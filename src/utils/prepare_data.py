# src/utils/prepare_data.py
from pathlib import Path
import pandas as pd

IN  = Path(r"c:\Users\tomas\Desktop\Data Science Project\data\processed\ff_size_deciles_with_ff3_monthly.csv")
OUT = Path(r"c:\Users\tomas\Desktop\Data Science Project\data\processed\ff_size_deciles_with_ff3_monthly_decimals.csv")

# --- 1) Load original and convert percent -> decimal ---------------------------------
df = pd.read_csv(IN, index_col=0)

# parse index as monthly period
try:
    df.index = pd.PeriodIndex(df.index.astype(str), freq="M")
except Exception:
    df.index = pd.to_datetime(df.index.astype(str), format="%Y-%m", errors="coerce")

df = df.apply(pd.to_numeric, errors="coerce") / 100.0  # percent -> decimal
df.index.name = "date"

# drop any stray 'Unnamed:*' cols
drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
if drop_cols:
    df = df.drop(columns=drop_cols)

# ---------- helpers ----------
def fmt_num(val, decimals=6, dec_char="."):
    if pd.isna(val):
        return ""
    s = f"{val:.{decimals}f}"
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s

# --- 2) Build aligned strings table for pretty CSV -----------------------------------
df_txt = df.reset_index().copy()
for c in df_txt.columns:
    if c != "date":
        df_txt[c] = df_txt[c].map(lambda v: fmt_num(v))

# compute column widths (max of header and values)
widths = {c: max(len(c), df_txt[c].astype(str).map(len).max()) for c in df_txt.columns}

# header (date left-aligned, numbers right-aligned)
header_cells = [f"{'date':<{widths['date']}}"] + [f"{c:>{widths[c]}}" for c in df_txt.columns if c != "date"]

# rows
row_lines = []
for _, row in df_txt.iterrows():
    left = f"{str(row['date']):<{widths['date']}}"
    nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "date"]
    row_lines.append([left] + nums)

# --- 3) Save one visually aligned CSV (semicolon separator) --------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
lines_csv = [";".join(header_cells)] + [";".join(cells) for cells in row_lines]
OUT.write_text("\n".join(lines_csv), encoding="utf-8-sig")
print("Saved visually aligned CSV →", OUT)
