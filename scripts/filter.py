import pandas as pd

df = pd.read_csv("data/reports/dataframe.csv")
df = df[df["sample_size"] > 0]
df.to_csv("dataframe_filtered.csv", index=False)
print(f"Rows remaining: {len(df)}")