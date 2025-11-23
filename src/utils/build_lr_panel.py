# src/utils/build_lr_panel.py
# Combine per-decile OOS *LR* predictions into wide panels for:
#   - RETURNS     (results/oos_preds_lr/ME*_oos_preds_lr.csv)
#   - VOLATILITY  (results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv)
#
# Outputs (LR only):
#   Return panels:
#       results/oos_panel_lr/preds_panel_lr.csv        (μ^LR: predicted returns)
#       results/oos_panel_lr/true_panel_lr.csv         (realized returns)
#   Volatility panels:
#       results/oos_vol_panel_lr/preds_vol_panel_lr.csv  (σ²_pred^LR: predicted squared returns)
#       results/oos_vol_panel_lr/true_vol_panel_lr.csv   (σ²_true: realized squared returns)

import os
import glob
from pathlib import Path

import pandas as pd

# Output directories for the final “panel” CSVs
RET_OUT_DIR = "results/oos_panel_lr"
VOL_OUT_DIR = "results/oos_vol_panel_lr"


# ---------- robust loader for aligned OOS CSVs (semicolon + padded) ----------

def _load_oos_csv(path: str) -> pd.DataFrame:
    """
    Load aligned OOS prediction CSVs written by:
      - train_all_lr.py        (returns)
      - train_all_vol_lr.py    (volatility)

    These files:
      - are typically semicolon-separated
      - may have padded columns for visual alignment
      - use dot decimals
      - have a 'month' column and numeric columns y_true, y_pred

    This function also works for plain CSVs without alignment.

    Args:
        path: path to a per-decile OOS CSV file.

    Returns:
        DataFrame with:
          - string 'month' column
          - numeric y_true / y_pred columns (when present).
    """
    # Peek at the first line to see if it's semicolon-separated
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        # Aligned semicolon-separated format
        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            dtype=str,
            encoding="utf-8-sig",
        )

        # Normalize header: strip whitespace + possible BOM char
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        # Strip spaces on all string columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        # Convert non-month columns to numeric (empty → NaN)
        for c in df.columns:
            if c != "month":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")

        return df
    else:
        # Fall back to a standard CSV
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------- pretty CSV writer (semicolon, aligned columns, trimmed zeros) ----------

def _fmt_num(val, max_decimals: int = 6, dec_char: str = "."):
    """
    Format a numeric value with trimmed trailing zeros, consistent with other scripts.

    Args:
        val: numeric value or NaN.
        max_decimals: maximum number of decimal places.
        dec_char: decimal separator ('.' by default).

    Returns:
        String representation, or "" if val is NaN.
    """
    if pd.isna(val):
        return ""
    s = f"{float(val):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s


def _write_aligned_csv(
    df: pd.DataFrame,
    path: str,
    max_decimals: int = 6,
    sep: str = ";",
) -> None:
    """
    Write a DataFrame as a semicolon-separated, visually aligned CSV.

    Rules:
        - 'month' column is left-aligned
        - all other columns are right-aligned if they look numeric
        - numeric cells are formatted via _fmt_num

    Args:
        df: DataFrame to write.
        path: output path.
        max_decimals: maximum decimals for numeric columns.
        sep: column separator (default ';').
    """
    df_str = df.copy()

    # Columns that should be left-aligned (only 'month' here)
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-looking columns on the right side
    for c in right_cols:
        ser_num = pd.to_numeric(df_str[c], errors="coerce")
        if ser_num.notna().any():
            # At least one numeric value: format all numeric entries
            df_str[c] = ser_num.map(
                lambda v: _fmt_num(v, max_decimals) if pd.notna(v) else ""
            )
        else:
            # Non-numeric column: keep as string
            df_str[c] = df_str[c].astype(str)

    # Make sure all values are strings before measuring widths
    df_str = df_str.astype(str)

    # Compute column widths as max between header length and value length
    widths = {
        c: max(len(c), df_str[c].map(len).max())
        for c in df_str.columns
    }

    # Build header row
    header_cells = []
    for c in df_str.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]

    # Build all data rows
    for _, row in df_str.iterrows():
        cells = []
        for c in df_str.columns:
            val = row[c]
            if c in left_cols:
                cells.append(f"{val:<{widths[c]}}")
            else:
                cells.append(f"{val:>{widths[c]}}")
        lines.append(sep.join(cells))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- RETURN PANELS (from results/oos_preds_lr) ----------

def build_return_panels() -> None:
    """
    Build wide LR return panels from per-decile OOS prediction files.

    Input:
        results/oos_preds_lr/ME*_oos_preds_lr.csv
            columns: month, y_true, y_pred

    Output:
        results/oos_panel_lr/preds_panel_lr.csv  (μ^LR: predicted returns, wide ME1..ME10)
        results/oos_panel_lr/true_panel_lr.csv   (realized returns, wide ME1..ME10)
    """
    files = sorted(glob.glob("results/oos_preds_lr/ME*_oos_preds_lr.csv"))
    if not files:
        raise FileNotFoundError(
            "No files found in results/oos_preds_lr/ME*_oos_preds_lr.csv"
        )

    # Collect all deciles in long format
    long_rows = []
    for p in files:
        # Extract decile name, e.g. "ME1" from "ME1_oos_preds_lr.csv"
        decile = os.path.basename(p).split("_")[0]

        df = _load_oos_csv(p)

        need = {"month", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"{p} missing columns {missing}")

        df = df[["month", "y_true", "y_pred"]].copy()
        df["decile"] = decile
        long_rows.append(df)

    # Stack everything into one long DataFrame
    long = pd.concat(long_rows, axis=0, ignore_index=True)

    # Enforce ME1 → ME10 order within each month
    long["dec_num"] = long["decile"].str.extract(r"ME(\d+)").astype(int)
    long = (
        long.sort_values(["month", "dec_num"])
            .drop(columns="dec_num")
            .reset_index(drop=True)
    )

    # Pivot to wide format:
    #   index  -> month
    #   columns -> decile (ME1..ME10)
    #   values  -> y_pred / y_true
    preds_wide = long.pivot(index="month", columns="decile", values="y_pred").sort_index()
    true_wide = long.pivot(index="month", columns="decile", values="y_true").sort_index()

    # Ensure column order ME1..ME10 (if some are missing, we keep the ones that exist)
    ordered_cols = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in preds_wide.columns]
    preds_wide = preds_wide[ordered_cols]
    true_wide = true_wide[ordered_cols]

    # Create output directory if needed
    os.makedirs(RET_OUT_DIR, exist_ok=True)

    # Reset index so 'month' becomes a column again before writing
    preds_out = preds_wide.reset_index()
    true_out = true_wide.reset_index()

    preds_path = os.path.join(RET_OUT_DIR, "preds_panel_lr.csv")
    true_path = os.path.join(RET_OUT_DIR, "true_panel_lr.csv")

    _write_aligned_csv(preds_out, preds_path, max_decimals=6)
    _write_aligned_csv(true_out, true_path, max_decimals=6)

    # Console summary
    print("Saved LR RETURN panels:")
    print(" -", preds_path)
    print(" -", true_path)
    print("Coverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(
        f" first month: {preds_wide.index.min()} | "
        f"last month: {preds_wide.index.max()}"
    )


# ---------- VOLATILITY PANELS (from results/oos_vol_preds_lr) ----------

def build_vol_panels() -> None:
    """
    Build wide LR volatility panels from per-decile OOS VOL prediction files.

    Input:
        results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv
            columns: month, y_true (squared return), y_pred (predicted squared return)

    Output:
        results/oos_vol_panel_lr/preds_vol_panel_lr.csv  (σ²_pred^LR: predicted squared returns)
        results/oos_vol_panel_lr/true_vol_panel_lr.csv   (σ²_true: realized squared returns)
    """
    files = sorted(glob.glob("results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv"))
    if not files:
        raise FileNotFoundError(
            "No files found in results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv"
        )

    # Collect all deciles in long format
    long_rows = []
    for p in files:
        # Extract decile name, e.g. "ME1" from "ME1_oos_vol_preds_lr.csv"
        decile = os.path.basename(p).split("_")[0]

        df = _load_oos_csv(p)

        need = {"month", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            missing = need - set(df.columns)
            raise ValueError(f"{p} missing columns {missing}")

        df = df[["month", "y_true", "y_pred"]].copy()
        df["decile"] = decile
        long_rows.append(df)

    # Stack everything
    long = pd.concat(long_rows, axis=0, ignore_index=True)

    # Enforce ME1 → ME10 order within each month
    long["dec_num"] = long["decile"].str.extract(r"ME(\d+)").astype(int)
    long = (
        long.sort_values(["month", "dec_num"])
            .drop(columns="dec_num")
            .reset_index(drop=True)
    )

    # Pivot to wide format
    preds_wide = long.pivot(index="month", columns="decile", values="y_pred").sort_index()
    true_wide = long.pivot(index="month", columns="decile", values="y_true").sort_index()

    # Ensure column order ME1..ME10
    ordered_cols = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in preds_wide.columns]
    preds_wide = preds_wide[ordered_cols]
    true_wide = true_wide[ordered_cols]

    # Create output directory
    os.makedirs(VOL_OUT_DIR, exist_ok=True)

    # Reset index so 'month' becomes a column again
    preds_out = preds_wide.reset_index()
    true_out = true_wide.reset_index()

    preds_path = os.path.join(VOL_OUT_DIR, "preds_vol_panel_lr.csv")
    true_path = os.path.join(VOL_OUT_DIR, "true_vol_panel_lr.csv")

    _write_aligned_csv(preds_out, preds_path, max_decimals=6)
    _write_aligned_csv(true_out, true_path, max_decimals=6)

    # Console summary
    print("\nSaved LR VOL panels:")
    print(" -", preds_path)
    print(" -", true_path)
    print("Coverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(
        f" first month: {preds_wide.index.min()} | "
        f"last month: {preds_wide.index.max()}"
    )


def main() -> None:
    """
    Build both LR return and LR volatility panels from per-decile OOS CSVs.
    """
    build_return_panels()
    build_vol_panels()


if __name__ == "__main__":
    main()
