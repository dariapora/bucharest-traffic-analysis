# usage: python3 classification.py input.csv

import sys
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score
)
import xgboost as xgb


def main():

    csv_path = sys.argv[1]
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows, {df['segment_id'].nunique():,} segments, {df['date'].nunique()} days")

    def label_3class(row):
        ratio = row["speed_ratio"]
        if ratio >= 0.75:
            return "free_flow"
        elif ratio >= 0.50:
            return "slow"
        else:
            return "congested"

    df["traffic_state_3"] = df.apply(label_3class, axis=1)

    state_order = ["free_flow", "slow", "congested"]
    state_to_num = {s: i for i, s in enumerate(state_order)}
    df["target"] = df["traffic_state_3"].map(state_to_num)

    print(f"\nTraffic state distribution:")
    for state in state_order:
        n = (df["traffic_state_3"] == state).sum()
        pct = n / len(df) * 100
        print(f"  {state:>12s}: {n:>10,} ({pct:.1f}%)")

    class_counts = df["target"].value_counts().sort_index()
    total = len(df)
    n_classes = len(state_order)
    class_weights = {i: total / (n_classes * class_counts[i]) for i in range(n_classes)}
    df["sample_weight"] = df["target"].map(class_weights)

    print(f"\nClass weights:")
    for i, state in enumerate(state_order):
        print(f"  {state:>12s}: {class_weights[i]:.3f}")

    seg_encoder = LabelEncoder()
    df["segment_id_enc"] = seg_encoder.fit_transform(df["segment_id"])

    df["time_sin"] = np.sin(2 * math.pi * df["time_numeric"] / 24)
    df["time_cos"] = np.cos(2 * math.pi * df["time_numeric"] / 24)

    feature_cols = [
        "segment_id_enc",   # which street
        "time_sin",         # time of day (cyclical component 1)
        "time_cos",         # time of day (cyclical component 2)
        "day_of_week_num",  # which day (0=Mon, 6=Sun)
        "is_weekend",       # weekday vs weekend
        "speed_limit",      # road speed limit
        "frc",              # road class
        "distance",         # segment length
    ]

    days = sorted(df["date"].unique())
    print(f"\nDays in dataset: {days}")
    print(f"Running leave-one-day-out cross-validation...\n")

    all_predictions = []

    for test_day in days:
        train = df[df["date"] != test_day]
        test = df[df["date"] == test_day]

        X_train = train[feature_cols]
        y_train = train["target"]
        w_train = train["sample_weight"]
        X_test = test[feature_cols]
        y_test = test["target"]

        model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=n_classes,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, sample_weight=w_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")

        day_name = test["day_of_week"].iloc[0]
        is_wknd = "weekend" if test["is_weekend"].iloc[0] == 1 else "weekday"
        print(f"  {test_day} ({day_name}, {is_wknd}): accuracy={acc:.4f}  F1-weighted={f1_w:.4f}  F1-macro={f1_m:.4f}")

        test_with_preds = test.copy()
        test_with_preds["predicted"] = preds
        test_with_preds["correct"] = (preds == y_test.values).astype(int)
        all_predictions.append(test_with_preds)

    results = pd.concat(all_predictions, ignore_index=True)
    overall_acc = results["correct"].mean()
    overall_f1_w = f1_score(results["target"], results["predicted"], average="weighted")
    overall_f1_m = f1_score(results["target"], results["predicted"], average="macro")

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY:    {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"OVERALL F1 WEIGHTED: {overall_f1_w:.4f}")
    print(f"OVERALL F1 MACRO:    {overall_f1_m:.4f}")
    print(f"{'='*60}")

    print(f"\nConfusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(results["target"], results["predicted"])
    cm_df = pd.DataFrame(cm, index=state_order, columns=state_order)
    print(cm_df.to_string())

    print(f"\nNormalized confusion matrix (% of each actual class):")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_norm_df = pd.DataFrame(
        np.round(cm_norm, 1),
        index=state_order, columns=state_order
    )
    print(cm_norm_df.to_string())

    print(f"\nClassification report:")
    print(classification_report(
        results["target"], results["predicted"],
        target_names=state_order, digits=4
    ))

    print(f"Feature importance (retrained on full dataset):")
    full_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softmax", num_class=n_classes,
        eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=-1,
    )
    full_model.fit(df[feature_cols], df["target"], sample_weight=df["sample_weight"])

    importances = full_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=False)

    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:>20s}: {row['importance']:.4f} {bar}")

    print(f"\nWeekday vs weekend:")
    for label, val in [("Weekday", 0), ("Weekend", 1)]:
        mask = results["is_weekend"] == val
        if mask.sum() > 0:
            acc = results.loc[mask, "correct"].mean()
            f1 = f1_score(
                results.loc[mask, "target"],
                results.loc[mask, "predicted"],
                average="macro"
            )
            print(f"  {label}: accuracy={acc:.4f} ({acc*100:.1f}%)  F1-macro={f1:.4f}")

    print(f"\nAccuracy by time of day (1-hour):")
    results["hour"] = results["time_numeric"].astype(int)
    for h in range(24):
        mask = results["hour"] == h
        if mask.sum() > 0:
            acc = results.loc[mask, "correct"].mean()
            print(f"  {h:02d}:00 - {h+1:02d}:00: {acc:.4f} ({acc*100:.1f}%)")

    seg_accuracy = results.groupby(["segment_id", "street_name", "latitude", "longitude"]).agg(
        accuracy=("correct", "mean"),
        num_observations=("correct", "count"),
    ).reset_index()

    seg_output = csv_path.replace(".csv", "_segment_accuracy.csv")
    seg_accuracy.to_csv(seg_output, index=False)
    print(f"\nPer-segment accuracy saved to {seg_output}")
    print(f"  ({len(seg_accuracy):,} segments)")
    print(f"  Most predictable:  {seg_accuracy['accuracy'].max():.4f}")
    print(f"  Least predictable: {seg_accuracy['accuracy'].min():.4f}")
    print(f"  Median accuracy:   {seg_accuracy['accuracy'].median():.4f}")


if __name__ == "__main__":
    main()