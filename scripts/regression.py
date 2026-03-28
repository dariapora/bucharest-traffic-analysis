
import pandas as pd
import numpy as np
import math
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("data/reports/dataframe.csv", sep=",")

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["day_of_week_num"]
df["is_weekend"] = df["is_weekend"].astype(int)

time_parts = df["time_start"].str.split(":", expand=True).astype(int)
df["hour"] = time_parts[0]
df["minute"] = time_parts[1]

df["hour_sin"]   = np.sin(2 * math.pi * df["hour"] / 24)
df["hour_cos"]   = np.cos(2 * math.pi * df["hour"] / 24)
df["minute_sin"] = np.sin(2 * math.pi * df["minute"] / 60)
df["minute_cos"] = np.cos(2 * math.pi * df["minute"] / 60)
df["dow_sin"]    = np.sin(2 * math.pi * df["day_of_week"] / 7)
df["dow_cos"]    = np.cos(2 * math.pi * df["day_of_week"] / 7)

features = [
    "hour_sin", "hour_cos", "minute_sin", "minute_cos",
    "dow_sin", "dow_cos", "is_weekend",
    "speed_limit", "frc", "distance",
    "latitude", "longitude",
    "sample_size",
    "seg_mean_speed", "seg_std_speed", "seg_median_speed",
    "seg_mean_ratio", "seg_mean_tt_ratio",
]
target = "average_speed"

df = df.dropna(subset=[target])
unique_dates = df["date"].unique()

results = []
for test_date in unique_dates:
    train = df[df["date"] != test_date].copy()
    test  = df[df["date"] == test_date].copy()

    seg_stats = train.groupby("segment_id").agg(
        seg_mean_speed=("average_speed", "mean"),
        seg_std_speed=("average_speed", "std"),
        seg_median_speed=("average_speed", "median"),
        seg_mean_ratio=("speed_ratio", "mean"),
        seg_mean_tt_ratio=("travel_time_ratio", "mean"),
    ).reset_index()

    train = train.merge(seg_stats, on="segment_id", how="left")
    test  = test.merge(seg_stats, on="segment_id", how="left")

    train = train.dropna(subset=features)
    test  = test.dropna(subset=features)

    if len(test) == 0:
        continue

    model = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
    )
    model.fit(train[features], train[target])

    preds = model.predict(test[features])
    test["predicted_speed"] = preds
    test["error"] = test["predicted_speed"] - test[target]


    print(test[["street_name", "time_start", "average_speed", "predicted_speed", "error"]].head(20))
    mae = mean_absolute_error(test[target], preds)
    r2  = r2_score(test[target], preds)
    results.append({"held_out_day": str(test_date)[:10], "MAE": mae, "R2": r2})
    print(f"  {str(test_date)[:10]}: MAE={mae:.2f}  R2={r2:.4f}")

res_df = pd.DataFrame(results)
print("\n" + res_df.to_string(index=False))
print(f"\nMean MAE: {res_df['MAE'].mean():.2f}")
print(f"Mean R2:  {res_df['R2'].mean():.4f}")