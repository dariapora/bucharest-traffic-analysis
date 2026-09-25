from pathlib import Path
import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "data" / "reports" / "dataframe.csv"

THRESHOLDS = {
    "free_flow": 0.85,
    "slow": 0.55
}

STATE_ORDER = [
    "free_flow",
    "slow",
    "congested"
]


def ratio_to_label(sr):
    if sr >= THRESHOLDS["free_flow"]:
        return 0       #free_flow

    elif sr >= THRESHOLDS["slow"]:
        return 1       #moderate

    else:
        return 2       #congested



def make_time_slot(df):
    return (df["time_numeric"] * 4).round().astype(int) % 96


def add_features(df, train_ref):
    df = df.copy()
    train_ref2 = train_ref.copy()
    train_ref2["hour"] = train_ref2["time_numeric"].astype(int)
    df["hour"] = df["time_numeric"].astype(int)

    # target-encode segment
    seg_enc = (
        train_ref.groupby("segment_id")["speed_ratio"]
        .agg(segment_mean_speed="mean", segment_std_speed="std",
             segment_min_speed="min", segment_q10_speed=lambda x: x.quantile(0.10))
        .fillna(0)
    )
    df = df.join(seg_enc, on="segment_id")

    # segment congestion rate (fraction of time in congested state)
    seg_cong = (
        (train_ref["speed_ratio"] < THRESHOLDS["slow"])
        .groupby(train_ref["segment_id"]).mean()
        .rename("segment_congestion_rate")
    )
    df = df.join(seg_cong, on="segment_id")

    # segment slow rate (fraction of time in slow state)
    seg_slow = (
        ((train_ref["speed_ratio"] >= THRESHOLDS["slow"]) &
         (train_ref["speed_ratio"] < THRESHOLDS["free_flow"]))
        .groupby(train_ref["segment_id"]).mean()
        .rename("segment_slow_rate")
    )
    df = df.join(seg_slow, on="segment_id")

    # historical aggregates
    for key, col in [
        (["segment_id", "day_of_week_num"], "hist_seg_dow"),
        (["segment_id", "is_weekend"],      "hist_seg_weekend"),
        (["segment_id", "hour"],            "hist_seg_hour"),
    ]:
        agg = train_ref2.groupby(key)["speed_ratio"].mean().rename(col)
        df = df.join(agg, on=key)

    # segment-hour std: how variable is this segment at this hour?
    hist_seg_hour_std = (
        train_ref2.groupby(["segment_id", "hour"])["speed_ratio"]
        .std().fillna(0).rename("hist_seg_hour_std")
    )
    df = df.join(hist_seg_hour_std, on=["segment_id", "hour"])

    # segment-hour congestion rate
    seg_hour_cong = (
        (train_ref2["speed_ratio"] < THRESHOLDS["slow"])
        .groupby([train_ref2["segment_id"], train_ref2["hour"]]).mean()
        .rename("seg_hour_cong_rate")
    )
    df = df.join(seg_hour_cong, on=["segment_id", "hour"])

    # segment-hour min and q25: worst-case behavior at this hour
    seg_hour_extremes = (
        train_ref2.groupby(["segment_id", "hour"])["speed_ratio"]
        .agg(seg_hour_min="min", seg_hour_q25=lambda x: x.quantile(0.25))
    )
    df = df.join(seg_hour_extremes, on=["segment_id", "hour"])

    global_hour = train_ref2.groupby("hour")["speed_ratio"].mean().rename("global_hour_mean")
    df = df.join(global_hour, on="hour")

    global_dow_hour = (
        train_ref2.groupby(["day_of_week_num", "hour"])["speed_ratio"]
        .mean().rename("global_dow_hour_mean")
    )
    df = df.join(global_dow_hour, on=["day_of_week_num", "hour"])

    # ── deviation features ────────────────────────────────────────────────────
    df["seg_vs_global_hour"] = df["hist_seg_hour"]  - df["global_hour_mean"]
    df["seg_vs_global_dow"]  = df["hist_seg_dow"]   - df["global_dow_hour_mean"]

    # ratio interaction: segment vs city at this hour
    df["seg_hour_vs_city_ratio"] = df["hist_seg_hour"] / (df["global_hour_mean"] + 1e-6)

    # ── road capacity proxy ───────────────────────────────────────────────────
    df["road_capacity"] = df["frc"] * df["speed_limit"]

    # ── peak hour flag (rush hours) ───────────────────────────────────────────
    df["is_peak"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # ── time features ─────────────────────────────────────────────────────────
    df["time_slot"] = make_time_slot(df)
    df["time_sin"]  = np.sin(2 * math.pi * df["time_numeric"] / 24)
    df["time_cos"]  = np.cos(2 * math.pi * df["time_numeric"] / 24)

    global_mean = train_ref["speed_ratio"].mean()
    agg_cols = [
        "segment_mean_speed", "segment_std_speed",
        "segment_min_speed", "segment_q10_speed",
        "segment_congestion_rate", "segment_slow_rate",
        "hist_seg_dow", "hist_seg_weekend", "hist_seg_hour",
        "hist_seg_hour_std",
        "seg_hour_cong_rate", "seg_hour_min", "seg_hour_q25",
        "global_hour_mean", "global_dow_hour_mean",
        "seg_vs_global_hour", "seg_vs_global_dow",
        "seg_hour_vs_city_ratio", "road_capacity"
    ]
    for col in agg_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].fillna(global_mean)

    return df


FEATURE_COLS = [

    "segment_mean_speed",
    "segment_std_speed",
    "segment_min_speed",
    "segment_q10_speed",

    "segment_congestion_rate",
    "segment_slow_rate",

    "seg_vs_global_hour",
    "seg_vs_global_dow",
    "seg_hour_vs_city_ratio",

    "hist_seg_hour_std",

    "seg_hour_cong_rate",
    "seg_hour_min",
    "seg_hour_q25",

    "speed_limit",
    "frc",
    "distance",
    "road_capacity",

    "is_peak",

    "time_slot",
    "time_sin",
    "time_cos",

    "day_of_week_num",
    "is_weekend",

    "hist_seg_dow",
    "hist_seg_weekend",
    "hist_seg_hour",

    "global_hour_mean",
    "global_dow_hour_mean",

    "sample_size"
]


df = pd.read_csv(csv_path)

print(
    f"Loaded {len(df):,} rows"
)

print(
    f"{df['segment_id'].nunique():,} segments"
)

print(
    f"{df['date'].nunique()} days"
)


df["target"] = (
    df["speed_ratio"]
    .apply(ratio_to_label)
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


    train = add_features(
        train,
        train
    )

    test = add_features(
        test,
        train
    )


    X_train = train[FEATURE_COLS]

    y_train = train["target"]

    X_test = test[FEATURE_COLS]

    y_test = test["target"]


    model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "logistic_regression",
            LogisticRegression(
                solver="saga",
                max_iter=500,
                class_weight="balanced",
                random_state=42
            )
        )

    ])


    model.fit(
        X_train,
        y_train
    )



    preds = model.predict(
        X_test
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
    "actual": y_test.to_numpy(),
    "predicted": preds
})

all_predictions.append(fold_results)


results = pd.concat(
    all_predictions,
    ignore_index=True
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