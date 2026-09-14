import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

from features import (
    REPORT_DIR,
    STATE_ORDER,
    FEATURE_COLS,
    load_dataset,
    add_features,
    add_features_out_of_fold,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_CLASSES = 3
MAX_ITER = 500


def compute_class_weights(y, n_classes):
    counts = np.bincount(y, minlength=n_classes)
    weights = len(y) / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def train_logistic_regression(X_train, y_train, n_classes=N_CLASSES, max_iter=MAX_ITER, seed=42):
    torch.manual_seed(seed)

    X = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y_train, dtype=torch.long, device=DEVICE)

    class_weights = compute_class_weights(y_train, n_classes).to(DEVICE)

    model = nn.Linear(X.shape[1], n_classes).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # full-batch L-BFGS: the problem is convex and fits on the GPU, so this
    # converges to the optimum in a few hundred evaluations
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        history_size=50,
        tolerance_grad=1e-7,
        tolerance_change=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        return loss

    model.train()
    optimizer.step(closure)

    return model


def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        preds = torch.argmax(model(X), dim=1)
    return preds.cpu().numpy()


df = load_dataset()

print(
    f"Loaded {len(df):,} rows"
)

print(
    f"{df['segment_id'].nunique():,} segments"
)

print(
    f"{df['date'].nunique()} days"
)


print("\nTraffic-state distribution:")

for i, state in enumerate(STATE_ORDER):

    count = (
        df["target"] == i
    ).sum()

    print(
        f"{state:>12s}: "
        f"{count:>10,} "
        f"({count / len(df) * 100:.1f}%)"
    )


days = sorted(
    df["date"]
    .unique()
)

all_predictions = []
day_metrics = []


for test_day in days:

    print(
        f"Testing day: {test_day}"
    )



    train = (
        df[df["date"] != test_day]
        .copy()
    )

    test = (
        df[df["date"] == test_day]
        .copy()
    )


    test = add_features(
        test,
        train
    )

    train = add_features_out_of_fold(
        train
    )


    X_train = train[FEATURE_COLS]

    y_train = train["target"]

    X_test = test[FEATURE_COLS]

    y_test = test["target"]


    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    model = train_logistic_regression(
        X_train_scaled,
        y_train.to_numpy()
    )



    preds = predict(
        model,
        X_test_scaled
    )


    acc = accuracy_score(
        y_test,
        preds
    )

    f1_w = f1_score(
        y_test,
        preds,
        average="weighted"
    )

    f1_m = f1_score(
        y_test,
        preds,
        average="macro"
    )


    print(
        f"  Accuracy:    {acc:.4f}"
    )

    print(
        f"  Weighted F1: {f1_w:.4f}"
    )

    print(
        f"  Macro F1:    {f1_m:.4f}"
    )


    day_metrics.append({
        "day": test_day,
        "accuracy": acc,
        "weighted_f1": f1_w,
        "macro_f1": f1_m
    })

    fold_results = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "time_numeric": test["time_numeric"].to_numpy(),
        "segment_id": test["segment_id"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": preds
    })

    all_predictions.append(fold_results)


results = pd.concat(
    all_predictions,
    ignore_index=True
)

results.to_parquet(
    REPORT_DIR / "predictions_logistic_regression.parquet",
    index=False
)


overall_accuracy = accuracy_score(
    results["actual"],
    results["predicted"]
)

overall_f1_weighted = f1_score(
    results["actual"],
    results["predicted"],
    average="weighted"
)

overall_f1_macro = f1_score(
    results["actual"],
    results["predicted"],
    average="macro"
)


print("\n")
print("=" * 60)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 60)

print(
    f"Overall accuracy:    "
    f"{overall_accuracy:.4f} "
    f"({overall_accuracy * 100:.2f}%)"
)

print(
    f"Weighted F1:         "
    f"{overall_f1_weighted:.4f}"
)

print(
    f"Macro F1:            "
    f"{overall_f1_macro:.4f}"
)


print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        results["actual"],
        results["predicted"],
        labels=[0, 1, 2],
        target_names=STATE_ORDER,
        digits=4
    )
)


cm = confusion_matrix(
    results["actual"],
    results["predicted"],
    labels=[0, 1, 2]
)


cm_df = pd.DataFrame(
    cm,
    index=STATE_ORDER,
    columns=STATE_ORDER
)


print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("Rows = actual")
print("Columns = predicted")
print("=" * 60)

print(
    cm_df.to_string()
)


cm_normalized = (
    cm.astype(float)
    /
    cm.sum(
        axis=1,
        keepdims=True
    )
)


cm_normalized_df = pd.DataFrame(
    cm_normalized,
    index=STATE_ORDER,
    columns=STATE_ORDER
)


print("\n")
print("=" * 60)
print("NORMALIZED CONFUSION MATRIX")
print("Rows sum to 100%")
print("=" * 60)

print(
    (
        cm_normalized_df * 100
    )
    .round(1)
    .to_string()
)


print("\n")
print("=" * 60)
print("PER-DAY RESULTS")
print("=" * 60)

day_metrics_df = pd.DataFrame(
    day_metrics
)

print(
    day_metrics_df.to_string(
        index=False
    )
)