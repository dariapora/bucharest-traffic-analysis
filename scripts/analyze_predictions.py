import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from features import REPORT_DIR, STATE_ORDER, LABEL_TAG, THRESHOLDS, MIN_PROBES, REFERENCE, load_dataset

MODELS = ["logistic_regression", "lightgbm", "xgboost", "xgboost_unweighted", "xgboost_with_sample_size"]


def metrics(actual, predicted):
    return dict(
        accuracy=accuracy_score(actual, predicted),
        weighted_f1=f1_score(actual, predicted, average="weighted"),
        macro_f1=f1_score(actual, predicted, average="macro"),
    )


def per_class_recall(actual, predicted):
    return {s: ((predicted == i) & (actual == i)).sum() / max((actual == i).sum(), 1)
            for i, s in enumerate(STATE_ORDER)}


def historical_mode_baseline(df):
    # non-ML reference: the segment's most frequent state in the same 15-min slot on the other days
    df = df[["date", "segment_id", "time_numeric", "target"]].copy()
    slot = (df["time_numeric"] * 4).round().astype(int)
    df["slot"] = slot
    preds = []
    for day in sorted(df["date"].unique()):
        ref = df[df["date"] != day]
        counts = (ref.groupby(["segment_id", "slot", "target"]).size()
                  .unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0))
        mode = counts.idxmax(axis=1).rename("predicted")
        test = df[df["date"] == day].join(mode, on=["segment_id", "slot"])
        test["predicted"] = test["predicted"].fillna(0).astype(int)
        preds.append(test[["date", "time_numeric", "segment_id", "target", "predicted"]]
                     .rename(columns={"target": "actual"}))
    return pd.concat(preds, ignore_index=True)


def main():
    df = load_dataset()
    day_type = df.groupby("date")["is_weekend"].first()
    lines = []
    p = lines.append

    runs = {}
    for m in MODELS:
        path = REPORT_DIR / f"predictions_{m}{LABEL_TAG}.parquet"
        if path.exists():
            runs[m] = pd.read_parquet(path)
    always_free = df[["date", "time_numeric", "segment_id", "target"]].rename(columns={"target": "actual"})
    always_free["predicted"] = 0
    runs = {"always_free_flow": always_free, "historical_mode": historical_mode_baseline(df), **runs}

    p(f"Label: reference={REFERENCE}, thresholds={THRESHOLDS}, min_probes={MIN_PROBES}, tag='{LABEL_TAG}'")
    p(f"Class distribution: {df['target'].value_counts(normalize=True).sort_index().round(4).tolist()}")
    p("")
    p("=" * 70)
    p("OVERALL (all 7 leave-one-day-out folds pooled)")
    p("=" * 70)
    rows = []
    for name, r in runs.items():
        mt = metrics(r["actual"], r["predicted"])
        rec = per_class_recall(r["actual"].to_numpy(), r["predicted"].to_numpy())
        rows.append(dict(model=name, **mt, **{f"recall_{k}": v for k, v in rec.items()}))
    p(pd.DataFrame(rows).round(4).to_string(index=False))

    p("")
    p("=" * 70)
    p("WEEKDAY vs WEEKEND (pooled folds)")
    p("=" * 70)
    rows = []
    for name, r in runs.items():
        wk = r["date"].map(day_type).astype(bool)
        for label, mask in [("weekday", ~wk), ("weekend", wk)]:
            rows.append(dict(model=name, days=label, **metrics(r.loc[mask, "actual"], r.loc[mask, "predicted"])))
    p(pd.DataFrame(rows).round(4).to_string(index=False))

    p("")
    p("=" * 70)
    p("ACCURACY BY HOUR OF DAY")
    p("=" * 70)
    hour_tab = {}
    for name, r in runs.items():
        h = r["time_numeric"].astype(int)
        hour_tab[name] = r.groupby(h).apply(lambda g: accuracy_score(g["actual"], g["predicted"]))
    hour_df = pd.DataFrame(hour_tab)
    ref = runs["xgboost"] if "xgboost" in runs else next(iter(runs.values()))
    hour_df["congested_share"] = ref.groupby(ref["time_numeric"].astype(int))["actual"].apply(lambda a: (a == 2).mean())
    hour_df["slow_share"] = ref.groupby(ref["time_numeric"].astype(int))["actual"].apply(lambda a: (a == 1).mean())
    p(hour_df.round(3).to_string())
    hour_df.round(4).to_csv(REPORT_DIR / f"accuracy_by_hour{LABEL_TAG}.csv")

    p("")
    p("=" * 70)
    p("MACRO-F1 BY HOUR OF DAY")
    p("=" * 70)
    f1_tab = {}
    for name, r in runs.items():
        h = r["time_numeric"].astype(int)
        f1_tab[name] = r.groupby(h).apply(lambda g: f1_score(g["actual"], g["predicted"], average="macro"))
    p(pd.DataFrame(f1_tab).round(3).to_string())

    p("")
    p("=" * 70)
    p("PER-SEGMENT ACCURACY (pooled folds)")
    p("=" * 70)
    seg_rows = {}
    for name, r in runs.items():
        seg = (r["actual"] == r["predicted"]).groupby(r["segment_id"]).mean()
        seg_rows[name] = seg
        q = seg.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3).to_dict()
        p(f"{name:>26s}: median={seg.median():.3f}  quantiles={q}  >=0.7: {(seg >= 0.7).mean():.3f}  <0.5: {(seg < 0.5).mean():.3f}")
    seg_df = pd.DataFrame(seg_rows)
    static = df.groupby("segment_id").agg(frc=("frc", "first"), speed_limit=("speed_limit", "first"),
                                          distance=("distance", "first"),
                                          latitude=("latitude", "first"), longitude=("longitude", "first"),
                                          mean_probes=("sample_size", "mean"),
                                          congested_share=("target", lambda t: (t == 2).mean()),
                                          street_name=("street_name", "first"))
    seg_df = seg_df.join(static)
    seg_df.to_csv(REPORT_DIR / f"segment_accuracy{LABEL_TAG}.csv")

    if "xgboost" in seg_df:
        p("")
        p("XGBoost per-segment accuracy by functional road class:")
        p(seg_df.groupby("frc")["xgboost"].agg(["count", "median", "mean"]).round(3).to_string())
        p("")
        p("XGBoost per-segment accuracy by segment congested share (how often the segment is congested):")
        b = pd.cut(seg_df["congested_share"], [-0.01, 0, 0.02, 0.05, 0.1, 0.2, 1.0])
        p(seg_df.groupby(b, observed=True)["xgboost"].agg(["count", "median", "mean"]).round(3).to_string())

    text = "\n".join(lines)
    print(text)
    (REPORT_DIR / f"analysis{LABEL_TAG}.md").write_text("# Cross-model analysis of saved predictions\n\n```\n" + text + "\n```\n",
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
