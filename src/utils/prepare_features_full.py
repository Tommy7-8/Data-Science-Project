# src/utils/prepare_features_full.py
import os
import argparse
import pandas as pd

def to_year_month(s):
    return pd.to_datetime(s).dt.to_period("M").astype(str)

def make_lags(series: pd.Series, max_lag: int):
    out = {}
    for k in range(1, max_lag + 1):
        out[f"{series.name}_lag{k}"] = series.shift(k)
    return pd.DataFrame(out)

def realized_vol(series: pd.Series, window: int):
    # population std of monthly returns
    return series.rolling(window=window, min_periods=window).std(ddof=0)

def build_features_for_decile(df: pd.DataFrame, decile: str) -> pd.DataFrame:
    # Keep date-like month and series we may need for checks
    out = df[["month", decile, "Mkt-RF", "SMB", "HML", "RF"]].copy()

    # Lags of the decile return
    out = pd.concat([out, make_lags(out[decile], 12)], axis=1)

    # Rolling realized vol of the decile
    out[f"{decile}_vol_3m"]  = realized_vol(out[decile], 3)
    out[f"{decile}_vol_6m"]  = realized_vol(out[decile], 6)
    out[f"{decile}_vol_12m"] = realized_vol(out[decile], 12)

    # Lagged common factors (only lagged are used in the model)
    out["Mkt-RF_lag1"] = out["Mkt-RF"].shift(1)
    out["SMB_lag1"]    = out["SMB"].shift(1)

    # Drop rows until all key lags/vols exist
    need = [f"{decile}_lag12", f"{decile}_vol_12m", "Mkt-RF_lag1", "SMB_lag1"]
    first_idx = out[need].dropna().index.min()
    out = out.loc[first_idx:].reset_index(drop=True)

    # month -> date (YYYY-MM)
    out = out.rename(columns={"month": "date"})
    return out

def main(args):
    p = args.input  # processed combined CSV (you said this lives in data/processed)
    df = pd.read_csv(p)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the CSV.")

    # Normalize to YYYY-MM and sort
    df["month"] = to_year_month(df["date"])
    df = df.sort_values("month").drop_duplicates(subset=["month"], keep="last").reset_index(drop=True)

    # Column checks
    needed = ["Mkt-RF", "SMB", "HML", "RF"] + [f"ME{i}" for i in range(1, 11)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    os.makedirs(args.outdir, exist_ok=True)

    # Build and write features per decile
    for j in range(1, 11):
        dec = f"ME{j}"
        feat = build_features_for_decile(df, dec)
        out_path = os.path.join(args.outdir, f"features_{dec}_full.csv")
        feat.to_csv(out_path, index=False)
        print(f"Wrote {out_path}  ({feat['date'].iloc[0]} → {feat['date'].iloc[-1]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv",
                        help="Path to the combined processed CSV.")
    parser.add_argument("--outdir", default="data/processed",
                        help="Directory to write features_ME*_full.csv files.")
    args = parser.parse_args()
    main(args)