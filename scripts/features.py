from pathlib import Path
import math
import os
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "reports" / "dataframe.csv"
REPORT_DIR = BASE_DIR / "data" / "reports" / "classification"

# label definition; overridable for sensitivity analysis
_thr = os.environ.get("TRAFFIC_THRESHOLDS", "0.85,0.55").split(",")
THRESHOLDS = {
    "free_flow": float(_thr[0]),
    "slow": float(_thr[1])
}
REFERENCE = os.environ.get("TRAFFIC_REFERENCE", "segment")   # "segment" (own p85 speed) or "limit" (posted limit)
LABEL_TAG = os.environ.get("TRAFFIC_LABEL_TAG", "")           # suffix for report files

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


MIN_PROBES = int(os.environ.get("TRAFFIC_MIN_PROBES", "3"))   # below this the interval is light traffic and its speed is noise
REF_MIN_PROBES = 5      # intervals used to estimate a segment's free-flow speed
REF_QUANTILE = 0.85


def free_flow_speed(df):
    # a segment's own free-flow speed instead of the posted limit: short
    # intersection segments and speed-bump streets never reach the limit even when empty
    reliable = df[df["sample_size"] >= REF_MIN_PROBES]
    ref = reliable.groupby("segment_id")["median_speed"].quantile(REF_QUANTILE)
    measured = df[df["sample_size"] > 0]
    fallback = measured.groupby("segment_id")["median_speed"].quantile(REF_QUANTILE)
    limit = df.groupby("segment_id")["speed_limit"].first()
    return ref.reindex(limit.index).fillna(fallback).fillna(limit)


def label_intervals(df):
    df = df.copy()
    if REFERENCE == "limit":
        df["free_flow_speed"] = df["speed_limit"].astype(float)
    else:
        df["free_flow_speed"] = df["segment_id"].map(free_flow_speed(df))
    df["speed_ratio"] = df["median_speed"] / df["free_flow_speed"]

    light = df["sample_size"] < MIN_PROBES
    df["target"] = df["speed_ratio"].apply(ratio_to_label)
    df.loc[light, "target"] = 0
    df.loc[light, "speed_ratio"] = np.nan
    return df


def make_time_slot(df):
    return (df["time_numeric"] * 4).round().astype(int) % 96


def add_features(df, train_ref):
    df = df.copy()
    train_ref2 = train_ref.copy()
    train_ref2["hour"] = train_ref2["time_numeric"].astype(int)
    df["hour"] = df["time_numeric"].astype(int)

    # target-encode segment
    seg_speed = train_ref.groupby("segment_id")["speed_ratio"]
    seg_enc = pd.DataFrame({
        "segment_mean_speed": seg_speed.mean(),
        "segment_std_speed": seg_speed.std(),
        "segment_min_speed": seg_speed.min(),
        "segment_q10_speed": seg_speed.quantile(0.10),
    }).fillna(0)
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
    seg_hour_speed = train_ref2.groupby(["segment_id", "hour"])["speed_ratio"]
    seg_hour_extremes = pd.DataFrame({
        "seg_hour_min": seg_hour_speed.min(),
        "seg_hour_q25": seg_hour_speed.quantile(0.25),
    })
    df = df.join(seg_hour_extremes, on=["segment_id", "hour"])

    # historical probe volume: how much traffic this segment normally carries at this hour
    seg_hour_probes = train_ref2.groupby(["segment_id", "hour"])["sample_size"]
    hist_probes = pd.DataFrame({
        "hist_seg_hour_probes": seg_hour_probes.mean(),
        "hist_seg_hour_light_rate": (train_ref2["sample_size"] < MIN_PROBES)
            .groupby([train_ref2["segment_id"], train_ref2["hour"]]).mean(),
    })
    df = df.join(hist_probes, on=["segment_id", "hour"])

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

    df["hist_seg_hour_probes"] = df["hist_seg_hour_probes"].fillna(train_ref["sample_size"].mean())
    df["hist_seg_hour_light_rate"] = df["hist_seg_hour_light_rate"].fillna((train_ref["sample_size"] < MIN_PROBES).mean())

    return df


def add_features_out_of_fold(train):
    # each training day gets aggregates computed from the other training days,
    # otherwise a row's own speed_ratio leaks into features like seg_hour_min
    parts = [
        add_features(train[train["date"] == d], train[train["date"] != d])
        for d in sorted(train["date"].unique())
    ]
    return pd.concat(parts)


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

    "hist_seg_hour_probes",
    "hist_seg_hour_light_rate"
]


def load_dataset():
    return label_intervals(pd.read_csv(CSV_PATH))
