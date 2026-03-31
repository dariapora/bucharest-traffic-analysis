# usage: python3 weights.py [data/dataframe.csv]
# XGBRegressor on speed_ratio → threshold to classes
# Produced: accuracy 73.9%, F1-macro 0.676

import sys
import math
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    f1_score, mean_absolute_error
)
import xgboost as xgb

# ── label definitions ─────────────────────────────────────────────────────────
# free_flow  : speed_ratio >= 0.70
# slow       : 0.40 <= speed_ratio < 0.70
# congested  : speed_ratio < 0.40
THRESHOLDS = {"free_flow": 0.85, "slow": 0.55}
STATE_ORDER = ["free_flow", "slow", "congested"]

def ratio_to_label(sr):
    if sr >= THRESHOLDS["free_flow"]:
        return 0   # free_flow
    elif sr >= THRESHOLDS["slow"]:
        return 1   # slow
    else:
        return 2   # congested

def ratio_to_label_custom(sr, t_free, t_slow):
    if sr >= t_free:
        return 0
    elif sr >= t_slow:
        return 1
    else:
        return 2

def optimize_thresholds(pred_ratios, true_labels, metric="accuracy"):
    """Grid-search for optimal (t_free, t_slow) on predicted ratios."""
    best_score = -1
    best_t = (THRESHOLDS["free_flow"], THRESHOLDS["slow"])
    # search around the default thresholds
    for t_free in np.arange(0.50, 0.85, 0.01):
        for t_slow in np.arange(0.20, t_free - 0.05, 0.01):
            preds = np.array([ratio_to_label_custom(r, t_free, t_slow) for r in pred_ratios])
            if metric == "accuracy":
                score = accuracy_score(true_labels, preds)
            else:
                score = f1_score(true_labels, preds, average="macro")
            if score > best_score:
                best_score = score
                best_t = (round(t_free, 2), round(t_slow, 2))
    return best_t, best_score

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

    # fill NaNs with global train mean
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
    "sample_size",
]

def evaluate_dataset(csv_path):
    """Run full LODO evaluation on a single dataset. Returns a dict of metrics."""
    label = csv_path.split("/")[-1].replace(".csv", "")
    print(f"\n{'#'*70}")
    print(f"# DATASET: {label}")
    print(f"# Path:    {csv_path}")
    print(f"{'#'*70}")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows, {df['segment_id'].nunique():,} segments, {df['date'].nunique()} days")

    # ── labels ────────────────────────────────────────────────────────────────
    df["target"] = df["speed_ratio"].apply(ratio_to_label)

    print(f"\nTraffic state distribution:")
    for i, state in enumerate(STATE_ORDER):
        n = (df["target"] == i).sum()
        print(f"  {state:>12s}: {n:>10,} ({n/len(df)*100:.1f}%)")

    # ── class weights ─────────────────────────────────────────────────────────
    class_counts = df["target"].value_counts().sort_index()
    total = len(df)
    n_classes = len(STATE_ORDER)
    class_weights = {i: total / (n_classes * class_counts[i]) for i in range(n_classes)}
    df["sample_weight"] = df["target"].map(class_weights)

    print(f"\nClass weights:")
    for i, state in enumerate(STATE_ORDER):
        print(f"  {state:>12s}: {class_weights[i]:.3f}")

    # ── LODO cross-validation ─────────────────────────────────────────────────
    days = sorted(df["date"].unique())
    print(f"\nDays in dataset: {days}")
    print(f"Running leave-one-day-out cross-validation...\n")
    all_predictions = []
    day_metrics = []

    for test_day in days:
        train = df[df["date"] != test_day].copy()
        test  = df[df["date"] == test_day].copy()

        train = add_features(train, train)
        test  = add_features(test,  train)

        X_train, y_train = train[FEATURE_COLS], train["speed_ratio"]
        w_train = train["sample_weight"]
        X_test,  y_test  = test[FEATURE_COLS],  test["target"]

        # ── XGBRegressor → threshold → class ──────────────────────────────────
        model = xgb.XGBRegressor(
            n_estimators=800,
            max_depth=5,
            learning_rate=0.02,
            min_child_weight=20,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric="rmse",
            verbosity=0,
            random_state=42,
            device="cuda",
            tree_method="hist",
        )
        model.fit(X_train, y_train, sample_weight=w_train)

        pred_ratios = model.predict(X_test)
        preds = np.array([ratio_to_label(r) for r in pred_ratios])

        acc  = accuracy_score(y_test, preds)
        f1_w = f1_score(y_test, preds, average="weighted")
        f1_m = f1_score(y_test, preds, average="macro")
        mae  = mean_absolute_error(test["speed_ratio"], pred_ratios)

        day_name = test["day_of_week"].iloc[0]
        is_wknd  = "weekend" if test["is_weekend"].iloc[0] == 1 else "weekday"
        print(f"  {test_day} ({day_name}, {is_wknd}): "
              f"acc={acc:.4f}  F1w={f1_w:.4f}  F1m={f1_m:.4f}  MAE(ratio)={mae:.4f}")
        day_metrics.append({
            "day": test_day, "day_name": day_name, "type": is_wknd,
            "acc": acc, "f1_w": f1_w, "f1_m": f1_m, "mae": mae,
        })

        test_with_preds = test.copy()
        test_with_preds["predicted"]  = preds
        test_with_preds["pred_ratio"] = pred_ratios
        test_with_preds["correct"]    = (preds == y_test.values).astype(int)
        all_predictions.append(test_with_preds)

    # ── aggregate results ─────────────────────────────────────────────────────
    results      = pd.concat(all_predictions, ignore_index=True)
    overall_acc  = results["correct"].mean()
    overall_f1_w = f1_score(results["target"], results["predicted"], average="weighted")
    overall_f1_m = f1_score(results["target"], results["predicted"], average="macro")
    overall_mae  = mean_absolute_error(results["speed_ratio"], results["pred_ratio"])

    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY:    {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"OVERALL F1 WEIGHTED: {overall_f1_w:.4f}")
    print(f"OVERALL F1 MACRO:    {overall_f1_m:.4f}")
    print(f"OVERALL MAE (ratio): {overall_mae:.4f}  <- continuous error")
    print(f"{'='*60}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(results["target"], results["predicted"])
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=STATE_ORDER, columns=STATE_ORDER).to_string())

    print(f"\nNormalized confusion matrix (%):")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print(pd.DataFrame(np.round(cm_norm, 1), index=STATE_ORDER, columns=STATE_ORDER).to_string())

    print(f"\nClassification report:")
    print(classification_report(results["target"], results["predicted"],
                                target_names=STATE_ORDER, digits=4))

    # ── feature importance ────────────────────────────────────────────────────
    print(f"Feature importance (retrained on full dataset):")
    full_df = add_features(df.copy(), df)
    full_model = xgb.XGBRegressor(
        n_estimators=800, max_depth=5, learning_rate=0.02,
        min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="rmse", verbosity=0, random_state=42,
        device="cuda", tree_method="hist",
    )
    full_model.fit(full_df[FEATURE_COLS], full_df["speed_ratio"],
                   sample_weight=full_df["sample_weight"])

    imp = full_model.feature_importances_
    imp = imp / imp.sum()
    for feat, val in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1]):
        bar = "\u2588" * int(val * 50)
        print(f"  {feat:>26s}: {val:.4f} {bar}")

    # ── weekday vs weekend ────────────────────────────────────────────────────
    wkday_acc = wkday_f1 = wkend_acc = wkend_f1 = None
    print(f"\nWeekday vs weekend:")
    for group_label, val in [("Weekday", 0), ("Weekend", 1)]:
        mask = results["is_weekend"] == val
        if mask.sum():
            acc_g = results.loc[mask, "correct"].mean()
            f1_g  = f1_score(results.loc[mask, "target"],
                             results.loc[mask, "predicted"], average="macro")
            print(f"  {group_label}: accuracy={acc_g:.4f}  F1-macro={f1_g:.4f}")
            if val == 0:
                wkday_acc, wkday_f1 = acc_g, f1_g
            else:
                wkend_acc, wkend_f1 = acc_g, f1_g

    # ── accuracy by hour ──────────────────────────────────────────────────────
    hour_accs = {}
    print(f"\nAccuracy by time of day (1-hour buckets):")
    results["hour"] = results["time_numeric"].astype(int)
    for h in range(24):
        mask = results["hour"] == h
        if mask.sum():
            a = results.loc[mask, "correct"].mean()
            hour_accs[h] = a
            bar = "\u2588" * int(a * 30)
            print(f"  {h:02d}:00  {a:.4f}  {bar}")

    # ── per-segment accuracy ──────────────────────────────────────────────────
    seg_accuracy = results.groupby(
        ["segment_id", "street_name", "latitude", "longitude"]
    ).agg(
        accuracy=("correct", "mean"),
        mean_pred_error=("pred_ratio", lambda x: (
            x - results.loc[x.index, "speed_ratio"]).abs().mean()),
        num_observations=("correct", "count"),
    ).reset_index()

    seg_output = csv_path.replace(".csv", "_segment_accuracy.csv")
    seg_accuracy.to_csv(seg_output, index=False)
    print(f"\nPer-segment accuracy saved to {seg_output}")
    print(f"  ({seg_accuracy['segment_id'].nunique():,} segments)")
    print(f"  Most predictable:  {seg_accuracy['accuracy'].max():.4f}")
    print(f"  Least predictable: {seg_accuracy['accuracy'].min():.4f}")
    print(f"  Median accuracy:   {seg_accuracy['accuracy'].median():.4f}")

    return {
        "label": label,
        "rows": len(df),
        "segments": df["segment_id"].nunique(),
        "class_dist": {s: (df["target"] == i).sum() for i, s in enumerate(STATE_ORDER)},
        "class_weights": class_weights,
        "overall_acc": overall_acc,
        "overall_f1_w": overall_f1_w,
        "overall_f1_m": overall_f1_m,
        "overall_mae": overall_mae,
        "day_metrics": day_metrics,
        "wkday_acc": wkday_acc, "wkday_f1": wkday_f1,
        "wkend_acc": wkend_acc, "wkend_f1": wkend_f1,
        "hour_accs": hour_accs,
        "confusion": confusion_matrix(results["target"], results["predicted"]),
    }


def print_comparison(r1, r2):
    """Print a side-by-side comparison of two evaluation runs."""
    print(f"\n{'='*80}")
    print(f"{'COMPARATIVE SUMMARY':^80}")
    print(f"{'='*80}")

    w = 30  # column width
    print(f"  {'Metric':<26s} {r1['label']:>{w}s} {r2['label']:>{w}s}   {'Delta':>8s}")
    print(f"  {'-'*26} {'-'*w} {'-'*w}   {'-'*8}")

    rows_data = [
        ("Rows",       f"{r1['rows']:,}",     f"{r2['rows']:,}",     None),
        ("Segments",   f"{r1['segments']:,}",  f"{r2['segments']:,}", None),
    ]
    for s in STATE_ORDER:
        n1, n2 = r1["class_dist"][s], r2["class_dist"][s]
        rows_data.append((f"  {s}", f"{n1:,} ({n1/r1['rows']*100:.1f}%)",
                          f"{n2:,} ({n2/r2['rows']*100:.1f}%)", None))
    for i, s in enumerate(STATE_ORDER):
        rows_data.append((f"  weight({s})",
                          f"{r1['class_weights'][i]:.3f}",
                          f"{r2['class_weights'][i]:.3f}", None))

    def delta_str(v1, v2):
        d = v2 - v1
        return f"{d:+.4f}"

    rows_data += [
        ("", "", "", None),
        ("Accuracy",   f"{r1['overall_acc']:.4f}",  f"{r2['overall_acc']:.4f}",
         delta_str(r1["overall_acc"], r2["overall_acc"])),
        ("F1 Weighted", f"{r1['overall_f1_w']:.4f}", f"{r2['overall_f1_w']:.4f}",
         delta_str(r1["overall_f1_w"], r2["overall_f1_w"])),
        ("F1 Macro",    f"{r1['overall_f1_m']:.4f}", f"{r2['overall_f1_m']:.4f}",
         delta_str(r1["overall_f1_m"], r2["overall_f1_m"])),
        ("MAE (ratio)", f"{r1['overall_mae']:.4f}",  f"{r2['overall_mae']:.4f}",
         delta_str(r1["overall_mae"], r2["overall_mae"])),
        ("", "", "", None),
        ("Weekday acc", f"{r1['wkday_acc']:.4f}" if r1["wkday_acc"] else "N/A",
         f"{r2['wkday_acc']:.4f}" if r2["wkday_acc"] else "N/A",
         delta_str(r1["wkday_acc"], r2["wkday_acc"]) if r1["wkday_acc"] and r2["wkday_acc"] else ""),
        ("Weekend acc", f"{r1['wkend_acc']:.4f}" if r1["wkend_acc"] else "N/A",
         f"{r2['wkend_acc']:.4f}" if r2["wkend_acc"] else "N/A",
         delta_str(r1["wkend_acc"], r2["wkend_acc"]) if r1["wkend_acc"] and r2["wkend_acc"] else ""),
        ("Weekday F1m", f"{r1['wkday_f1']:.4f}" if r1["wkday_f1"] else "N/A",
         f"{r2['wkday_f1']:.4f}" if r2["wkday_f1"] else "N/A",
         delta_str(r1["wkday_f1"], r2["wkday_f1"]) if r1["wkday_f1"] and r2["wkday_f1"] else ""),
        ("Weekend F1m", f"{r1['wkend_f1']:.4f}" if r1["wkend_f1"] else "N/A",
         f"{r2['wkend_f1']:.4f}" if r2["wkend_f1"] else "N/A",
         delta_str(r1["wkend_f1"], r2["wkend_f1"]) if r1["wkend_f1"] and r2["wkend_f1"] else ""),
    ]

    for metric, v1, v2, d in rows_data:
        d_str = f"   {d}" if d else ""
        print(f"  {metric:<26s} {v1:>{w}s} {v2:>{w}s}{d_str}")

    # ── per-day comparison ────────────────────────────────────────────────────
    print(f"\n  Per-day accuracy comparison:")
    print(f"  {'Day':<14s} {'Type':<8s} {r1['label']:>{w}s} {r2['label']:>{w}s}   {'Delta':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*w} {'-'*w}   {'-'*8}")
    for d1, d2 in zip(r1["day_metrics"], r2["day_metrics"]):
        d = d2["acc"] - d1["acc"]
        print(f"  {d1['day']:<14s} {d1['type']:<8s} "
              f"{d1['acc']:>{w}.4f} {d2['acc']:>{w}.4f}   {d:+.4f}")

    # ── per-hour comparison ───────────────────────────────────────────────────
    print(f"\n  Hourly accuracy comparison:")
    print(f"  {'Hour':<6s} {r1['label']:>14s} {r2['label']:>14s}   {'Delta':>8s}  {'Better':>10s}")
    print(f"  {'-'*6} {'-'*14} {'-'*14}   {'-'*8}  {'-'*10}")
    for h in range(24):
        a1 = r1["hour_accs"].get(h)
        a2 = r2["hour_accs"].get(h)
        if a1 is not None and a2 is not None:
            d = a2 - a1
            better = r2["label"] if d > 0.001 else (r1["label"] if d < -0.001 else "tie")
            print(f"  {h:02d}:00  {a1:14.4f} {a2:14.4f}   {d:+.4f}  {better:>10s}")

    print(f"\n{'='*80}")


def main():
    datasets = [
        "data/dataframe.csv",
        # "data/dataframe_positive_samples.csv",
    ]

    results_all = []
    for csv_path in datasets:
        r = evaluate_dataset(csv_path)
        results_all.append(r)

    print_comparison(results_all[0], results_all[1])

if __name__ == "__main__":
    main()