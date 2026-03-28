import json
import pandas as pd
import numpy as np
from pathlib import Path
import math
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def parse_tomtom_json(filepath):
    with open(filepath) as f:
        data = json.load(f)


    timeset_lookup = {ts["@id"]: ts["name"] for ts in data["timeSets"]}

    date_str = data["dateRanges"][0]["from"]


    rows = []
    for seg in data["network"]["segmentResults"]:
        seg_id = seg["segmentId"]
        street = seg.get("streetName", "")
        speed_limit = seg.get("speedLimit")
        frc = seg.get("frc")
        distance = seg.get("distance")


        for result in seg["segmentTimeResults"]:
            ts_id = result["timeSet"]
            interval_name = timeset_lookup[ts_id]  # e.g. "18:00-18:15"
            hour, minute = map(int, interval_name.split("-")[0].split(":"))


            rows.append({
                "date": date_str,
                "segment_id": seg_id,
                "street_name": street,
                "speed_limit": speed_limit,
                "frc": frc,
                "distance": distance,
                "interval": interval_name,
                "hour": hour,
                "minute": minute,
                "average_speed": result.get("averageSpeed"),
                "median_speed": result.get("medianSpeed"),
                "sample_size": result.get("sampleSize"),
            })


    return pd.DataFrame(rows)


all_dfs = [parse_tomtom_json(p) for p in Path("data/reports/json/").glob("*.json")]
df = pd.concat(all_dfs, ignore_index=True)



df["hour_sin"] = np.sin(2 * math.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * math.pi * df["hour"] / 24)
df["minute_sin"] = np.sin(2 * math.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * math.pi * df["minute"] / 60)


df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek      # 0=Monday
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)


df["dow_sin"] = np.sin(2 * math.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * math.pi * df["day_of_week"] / 7)


df["segment_id_enc"] = df["segment_id"].astype("category").cat.codes




features = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin", "dow_cos", "is_weekend",
    "speed_limit", "frc", "distance", "segment_id_enc"
]
target = "average_speed"


df = df.dropna(subset=features + [target])
unique_dates = df["date"].unique()


results = []
for test_date in unique_dates:
    train = df[df["date"] != test_date]
    test  = df[df["date"] == test_date]


    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
    model.fit(train[features], train[target])


    preds = model.predict(test[features])
    mae = mean_absolute_error(test[target], preds)
    r2  = r2_score(test[target], preds)
    results.append({"held_out_day": str(test_date)[:10], "MAE": mae, "R2": r2})


print(pd.DataFrame(results))