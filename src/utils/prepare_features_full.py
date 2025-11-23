# src/utils/prepare_features_full.py

import os
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Small helpers for date handling and basic feature construction
# ---------------------------------------------------------------------------

def to_year_month(s):
    """
    Convert a date-like Series to 'YYYY-MM' strings (monthly period).

    This is used to normalize whatever date format is in the CSV
    into a consistent monthly representation.
    """
    return pd.to_datetime(s).dt.to_period("M").astype(str)


def make_lags(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """
    Build a DataFrame of lagged versions of a single time series.

    Example:
        make_lags(ret, 3) will create:
            ret_lag1, ret_lag2, ret_lag3

    Args:
        series: pd.Series with a name attribute, e.g. 'ME1'.
        max_lag: maximum lag to create (starting from 1).

    Returns:
        DataFrame with one column per lag.
    """
    out = {}
    for k in range(1, max_lag + 1):
        out[f"{series.name}_lag{k}"] = series.shift(k)
    return pd.DataFrame(out)


def realized_vol(series: pd.Series, window: int) -> pd.Series:
    """
    Compute a simple rolling (realized) volatility as a rolling std dev.

    Args:
        series: return series (e.g. ME1).
        window: number of periods in the rolling window.

    Returns:
        Series of rolling standard deviations (ddof=0).
    """
    return series.rolling(window=window, min_periods=window).std(ddof=0)


# ---------------------------------------------------------------------------
# Robust loader for aligned CSV (supports ; separator + padded cells)
# ---------------------------------------------------------------------------

def load_processed_csv(path: str) -> pd.DataFrame:
    """
    Load a processed CSV in a way that is robust to:
        - semicolon-separated files with padded columns (as produced by our
          aligned writers)
        - UTF-8 with BOM
        - dot or comma as decimal separator

    If the first line contains a ';', we treat it as the "aligned" format
    written by download/feature scripts. Otherwise we fall back to a plain
    pd.read_csv.

    Args:
        path: path to the CSV file.

    Returns:
        DataFrame with numeric columns parsed where possible.
    """
    # Peek at the first line to see if the file is semicolon-separated.
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    # Case 1: "aligned" semicolon-separated CSV
    if ";" in head:
        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            dtype=str,
            encoding="utf-8-sig",
        )

        # Strip whitespace and BOM remnants from column names
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        # Strip whitespace from all string cells
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Convert non-date columns to numeric.
        # This handles both comma and dot decimals.
        for c in df.columns:
            if c != "date":
                s = df[c].replace("", pd.NA).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(s, errors="coerce")

        return df

    # Case 2: plain CSV with default delimiter
    else:
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Pretty CSV writer (semicolon, dot decimals, aligned columns)
# ---------------------------------------------------------------------------

def fmt_num(val, max_decimals=6, dec_char="."):
    """
    Match download_ff.py: format numbers without unnecessary trailing zeros.

    Args:
        val: numeric value (or NaN).
        max_decimals: maximum decimals to display.
        dec_char: decimal separator ('.' by default).

    Returns:
        Formatted string, or empty string for NaN.
    """
    if pd.isna(val):
        return ""
    # Fixed precision first
    s = f"{val:.{max_decimals}f}"
    # Trim trailing zeros and possible trailing dot
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    # Replace decimal separator if requested
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s


def write_aligned_csv(df: pd.DataFrame, path: str, decimals: int = 6) -> None:
    """
    Write a DataFrame as a semicolon-separated CSV with aligned columns.

    - 'date' column is left-aligned
    - numeric columns are right-aligned
    - decimals are formatted via fmt_num

    This is mainly for nice human-readable inspection; it still parses fine.

    Args:
        df: DataFrame with a 'date' column.
        path: output file path.
        decimals: maximum decimals for numeric columns.
    """
    df_txt = df.copy()

    # Format numeric columns (anything except 'date')
    for c in df_txt.columns:
        if c != "date":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v, decimals))

    # Compute column widths from max(header_length, max_value_length)
    widths = {
        c: max(len(c), df_txt[c].astype(str).map(len).max())
        for c in df_txt.columns
    }

    # Header: date left-aligned, others right-aligned
    header_cells = [f"{'date':<{widths['date']}}"] + [
        f"{c:>{widths[c]}}" for c in df_txt.columns if c != "date"
    ]

    lines = [";".join(header_cells)]

    # Data rows
    for _, row in df_txt.iterrows():
        left = f"{str(row['date']):<{widths['date']}}"
        nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "date"]
        lines.append(";".join([left] + nums))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Feature construction for a single size decile
# ---------------------------------------------------------------------------

def build_features_for_decile(df: pd.DataFrame, decile: str) -> pd.DataFrame:
    """
    Build the full feature set for a single ME decile (e.g. 'ME1').

    Features include:
        - Target return: decile
        - Factors: Mkt-RF, SMB, HML, RF
        - Lags of decile returns: decile_lag1..decile_lag12
        - Realized volatilities: decile_vol_3m, decile_vol_6m, decile_vol_12m
        - 1-month lags of Mkt-RF and SMB

    The function trims the initial rows until all required features exist
    (no NaNs in the last lag/vol and factor lags).

    Args:
        df: input DataFrame, must contain:
            ['month', decile, 'Mkt-RF', 'SMB', 'HML', 'RF']
        decile: column name of the decile (e.g. 'ME1').

    Returns:
        Feature DataFrame with a 'date' column (string YYYY-MM) and all
        the above features, starting from the first fully-available month.
    """
    # Start from month, target decile, and Fama–French factors
    out = df[["month", decile, "Mkt-RF", "SMB", "HML", "RF"]].copy()

    # Add lags of the decile returns
    out = pd.concat([out, make_lags(out[decile], 12)], axis=1)

    # Add realized volatility features
    out[f"{decile}_vol_3m"] = realized_vol(out[decile], 3)
    out[f"{decile}_vol_6m"] = realized_vol(out[decile], 6)
    out[f"{decile}_vol_12m"] = realized_vol(out[decile], 12)

    # Add 1-month lags of selected factors
    out["Mkt-RF_lag1"] = out["Mkt-RF"].shift(1)
    out["SMB_lag1"] = out["SMB"].shift(1)

    # We require at least:
    #   - 12-month lag of the decile
    #   - 12-month vol
    #   - lagged factors
    need = [f"{decile}_lag12", f"{decile}_vol_12m", "Mkt-RF_lag1", "SMB_lag1"]

    # Find the first row where all required features are non-NaN
    first_idx = out[need].dropna().index.min()

    # Drop everything before that row
    out = out.loc[first_idx:].reset_index(drop=True)

    # Rename 'month' to the generic 'date' for downstream consistency
    out = out.rename(columns={"month": "date"})
    return out


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

def main(args):
    """
    Read the combined Fama–French size-decile + FF3 CSV, build features for
    ME1..ME10, and write one aligned CSV per decile.

    Args:
        args: parsed command-line arguments with:
              - args.input: path to processed combined CSV
              - args.outdir: directory where features_ME*_full.csv are written
    """
    p = args.input
    df = load_processed_csv(p)

    # ------------------------------------------------------------------
    # Normalize / construct 'month' column
    # ------------------------------------------------------------------
    if "date" in df.columns:
        # Typical case: 'date' already exists as column
        df["month"] = to_year_month(df["date"])
    else:
        # Some files may have the date as the first unnamed column
        first_col = df.columns[0]
        # We only treat it as 'date' if it already looks like YYYY-MM
        if df[first_col].astype(str).str.match(r"\d{4}-\d{2}$").all():
            df = df.rename(columns={first_col: "date"})
            df["month"] = to_year_month(df["date"])
        else:
            raise ValueError("Expected a 'date' column in the CSV.")

    # Sort by month, keep last record per month (if duplicates), and reset index
    df = (
        df.sort_values("month")
          .drop_duplicates(subset=["month"], keep="last")
          .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Sanity checks: make sure all needed columns are present
    # ------------------------------------------------------------------
    needed = ["Mkt-RF", "SMB", "HML", "RF"] + [f"ME{i}" for i in range(1, 11)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure output directory exists
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build and save features for each size decile ME1..ME10
    # ------------------------------------------------------------------
    for j in range(1, 11):
        dec = f"ME{j}"
        feat = build_features_for_decile(df, dec)

        out_path = os.path.join(args.outdir, f"features_{dec}_full.csv")
        write_aligned_csv(feat, out_path, decimals=6)

        print(
            f"Wrote {out_path}  "
            f"({feat['date'].iloc[0]} → {feat['date'].iloc[-1]})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv",
        help="Path to the combined processed CSV.",
    )
    parser.add_argument(
        "--outdir",
        default="data/processed",
        help="Directory to write features_ME*_full.csv files.",
    )
    args = parser.parse_args()
    main(args)
