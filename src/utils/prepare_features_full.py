# src/utils/prepare_features_full.py
import os
import argparse
from pathlib import Path
import pandas as pd

def to_year_month(s):
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def make_lags(series: pd.Series, max_lag: int):
    out = {}
    for k in range(1, max_lag + 1):
        out[f"{series.name}_lag{k}"] = series.shift(k)
    return pd.DataFrame(out)

def realized_vol(series: pd.Series, window: int):
    return series.rolling(window=window, min_periods=window).std(ddof=0)

# ---------- robust loader for aligned CSV (handles ; + padded cells) ----------
def load_processed_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", engine="python", dtype=str, encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        # convert numeric columns (supports either comma or dot decimals)
        for c in df.columns:
            if c != "date":
                s = df[c].replace("", pd.NA).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(s, errors="coerce")
        return df
    else:
        return pd.read_csv(path, encoding="utf-8-sig")

# ---------- pretty CSV writer (semicolon, dot decimals, aligned columns) ----------
def fmt_num(val, max_decimals=6, dec_char="."):
    """Match download_ff.py: format without unnecessary trailing zeros."""
    if pd.isna(val):
        return ""
    s = f"{val:.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s

def write_aligned_csv(df: pd.DataFrame, path: str, decimals: int = 6):
    df_txt = df.copy()
    for c in df_txt.columns:
        if c != "date":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v, decimals))

    widths = {c: max(len(c), df_txt[c].astype(str).map(len).max()) for c in df_txt.columns}

    header_cells = [f"{'date':<{widths['date']}}"] + [
        f"{c:>{widths[c]}}" for c in df_txt.columns if c != "date"
    ]

    lines = [";".join(header_cells)]
    for _, row in df_txt.iterrows():
        left = f"{str(row['date']):<{widths['date']}}"
        nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "date"]
        lines.append(";".join([left] + nums))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")

def build_features_for_decile(df: pd.DataFrame, decile: str) -> pd.DataFrame:
    out = df[["month", decile, "Mkt-RF", "SMB", "HML", "RF"]].copy()
    out = pd.concat([out, make_lags(out[decile], 12)], axis=1)
    out[f"{decile}_vol_3m"]  = realized_vol(out[decile], 3)
    out[f"{decile}_vol_6m"]  = realized_vol(out[decile], 6)
    out[f"{decile}_vol_12m"] = realized_vol(out[decile], 12)
    out["Mkt-RF_lag1"] = out["Mkt-RF"].shift(1)
    out["SMB_lag1"]    = out["SMB"].shift(1)
    need = [f"{decile}_lag12", f"{decile}_vol_12m", "Mkt-RF_lag1", "SMB_lag1"]
    first_idx = out[need].dropna().index.min()
    out = out.loc[first_idx:].reset_index(drop=True)
    out = out.rename(columns={"month": "date"})
    return out

def main(args):
    p = args.input
    df = load_processed_csv(p)

    if "date" in df.columns:
        df["month"] = to_year_month(df["date"])
    else:
        first_col = df.columns[0]
        if df[first_col].astype(str).str.match(r"\d{4}-\d{2}$").all():
            df = df.rename(columns={first_col: "date"})
            df["month"] = to_year_month(df["date"])
        else:
            raise ValueError("Expected a 'date' column in the CSV.")

    df = df.sort_values("month").drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)

    needed = ["Mkt-RF", "SMB", "HML", "RF"] + [f"ME{i}" for i in range(1, 11)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    os.makedirs(args.outdir, exist_ok=True)

    for j in range(1, 11):
        dec = f"ME{j}"
        feat = build_features_for_decile(df, dec)
        out_path = os.path.join(args.outdir, f"features_{dec}_full.csv")
        write_aligned_csv(feat, out_path, decimals=6)
        print(f"Wrote {out_path}  ({feat['date'].iloc[0]} → {feat['date'].iloc[-1]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv",
                        help="Path to the combined processed CSV.")
    parser.add_argument("--outdir", default="data/processed",
                        help="Directory to write features_ME*_full.csv files.")
    args = parser.parse_args()
    main(args)
