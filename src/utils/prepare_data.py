from pathlib import Path
import pandas as pd

IN = Path(r"c:\Users\tomas\Desktop\Data Science Project\data\processed\ff_size_deciles_with_ff3_monthly.csv")
OUT = Path(r"c:\Users\tomas\Desktop\Data Science Project\data\processed\ff_size_deciles_with_ff3_monthly_decimals.csv")

# load and use first column as index
df = pd.read_csv(IN, index_col=0)

# parse index as monthly period (keeps dates as index, not a column)
try:
    df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
except Exception:
    df.index = pd.to_datetime(df.index.astype(str), format="%Y-%m", errors="coerce")

# convert percent -> decimal
df = df.apply(pd.to_numeric, errors="coerce") / 100.0

# ensure the index will be written with a proper header instead of "Unnamed: 0"
df.index.name = "date"

# optional: if a stray 'Unnamed: 0' column still exists, drop it
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# save
df.to_csv(OUT)  # index name 'date' will be used as the first column header
print("Saved decimal file without 'Unnamed: 0' to:", OUT)