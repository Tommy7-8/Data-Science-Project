# src/utils/build_gb_panels.py
# Collect per-decile GB OOS predictions into two panels:
#   - GB returns panel      (predicted returns only)
#   - GB volatility panel   (predicted volatility only)
#
# Inputs (from GB training scripts, aligned ; CSVs):
#   results/oos_preds_gb/ME*_oos_preds_gb.csv
#   results/oos_vol_gb/ME*_oos_vol_preds_gb.csv
#
# Outputs (predicted panels only, wide format month × deciles):
#   results/oos_panel_gb/preds_panel_gb.csv
#   results/oos_vol_panel_gb/preds_vol_panel_gb.csv

import os
import glob
from pathlib import Path

import pandas as pd

# We always work with the 10 Fama–French size deciles: ME1..ME10
DECILES = [f"ME{i}" for i in range(1, 11)]

BASE_RESULTS = Path("results")

# Per-decile GB OOS prediction directories
RET_PRED_DIR = BASE_RESULTS / "oos_preds_gb"
VOL_PRED_DIR = BASE_RESULTS / "oos_vol_gb"

# Output directories for panels
RET_OUT_DIR = BASE_RESULTS / "oos_panel_gb"
VOL_OUT_DIR = BASE_RESULTS / "oos_vol_panel_gb"

# Final wide panel CSV paths
RET_OUT_FILE = RET_OUT_DIR / "preds_panel_gb.csv"
VOL_OUT_FILE = VOL_OUT_DIR / "preds_vol_panel_gb.csv"


# ---------- robust loader for aligned OOS CSVs (semicolon + padded) ----------

def _load_oos_csv(path: Path) -> pd.DataFrame:
    """
    Load a single GB OOS prediction CSV.

    Files can be:
      - our aligned format:
          * semicolon-separated
          * padded cells for visual alignment
      - or a plain comma-separated CSV.

    In both cases the result will have at least:
      - 'month' column
      - 'y_pred' column (and usually 'y_true')

    Args:
        path: path to the OOS CSV.

    Returns:
        DataFrame with 'month' as string and numeric columns where possible.
    """
    # Peek at the first line to detect semicolon-separated format
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(
            path,
            sep=";",
            engine="python",
            dtype=str,
            encoding="utf-8-sig",
        )

        # Normalize header: strip whitespace and potential BOM char
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        # Strip spaces on object columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        # Convert non-month columns to numeric
        for c in df.columns:
            if c != "month":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")

        return df

    # Fall back to plain CSV
    return pd.read_csv(path, encoding="utf-8-sig")


# ---------- pretty CSV writer (semicolon, aligned columns, trimmed zeros) ----------

def _fmt_num(val, max_decimals: int = 6, dec_char: str = ".") -> str:
    """
    Format a numeric value with trimmed trailing zeros.

    This is consistent with the formatting used in the other scripts,
    so that all result CSVs look and parse the same.

    Args:
        val: numeric value or NaN.
        max_decimals: maximum number of decimal places.
        dec_char: decimal separator ('.' by default).

    Returns:
        String representation or "" if val is NaN.
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
    path: Path,
    max_decimals: int = 6,
    sep: str = ";",
) -> None:
    """
    Write a DataFrame as a semicolon-separated, visually aligned CSV.

    Rules:
      - 'month' column (if present) is left-aligned
      - all other columns are right-aligned
      - numeric-looking columns are formatted using _fmt_num

    Args:
        df: DataFrame to write.
        path: output file path.
        max_decimals: maximum decimals for numeric columns.
        sep: column separator (default ';').
    """
    df_txt = df.copy()

    # Columns to left-align (only 'month' here)
    left_cols = [c for c in df_txt.columns if c == "month"]
    right_cols = [c for c in df_txt.columns if c not in left_cols]

    # Format numeric-looking columns
    for c in right_cols:
        ser_num = pd.to_numeric(df_txt[c], errors="coerce")
        if ser_num.notna().any():
            df_txt[c] = ser_num.map(
                lambda v: _fmt_num(v, max_decimals) if pd.notna(v) else ""
            )
        else:
            # Non-numeric: keep as string
            df_txt[c] = df_txt[c].astype(str)

    # Convert everything to string before computing column widths
    df_txt = df_txt.astype(str)
    widths = {
        c: max(len(c), df_txt[c].map(len).max())
        for c in df_txt.columns
    }

    # Build header row
    header_cells = []
    for c in df_txt.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]

    # Build data rows
    for _, row in df_txt.iterrows():
        cells = []
        for c in df_txt.columns:
            val = row[c]
            if c in left_cols:
                cells.append(f"{val:<{widths[c]}}")
            else:
                cells.append(f"{val:>{widths[c]}}")
        lines.append(sep.join(cells))

    path.write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- generic loader: from MEj OOS to wide panel of y_pred ----------

def _load_pred_panel(pred_dir: Path, suffix: str) -> pd.DataFrame:
    """
    Build a wide (month × decile) panel from per-decile GB OOS prediction files.

    For each decile in DECILES:
      - read `<pred_dir>/<decile><suffix>`
      - take the 'y_pred' column and index it by 'month'
      - stack all deciles horizontally into a single DataFrame

    Args:
        pred_dir: directory containing per-decile prediction CSVs.
        suffix: filename suffix, e.g. "_oos_preds_gb.csv" or "_oos_vol_preds_gb.csv".

    Returns:
        DataFrame with:
          - index: 'month' (string YYYY-MM)
          - columns: decile names "ME1".. "ME10" (for those that exist).
    """
    series_list = []

    for dec in DECILES:
        path = pred_dir / f"{dec}{suffix}"
        if not path.exists():
            # We don't fail if a decile is missing; just warn and continue.
            print(f"WARNING: missing {path}, skipping {dec}")
            continue

        df = _load_oos_csv(path)

        if "month" not in df.columns:
            raise ValueError(f"'month' column not found in {path}")
        if "y_pred" not in df.columns:
            raise ValueError(f"'y_pred' column not found in {path}")

        # Keep predicted values with 'month' as index
        s = pd.Series(
            df["y_pred"].values,
            index=df["month"].astype(str),
            name=dec,
        )
        series_list.append(s)

    if not series_list:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    # Combine deciles into a wide panel and sort by month
    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index.name = "month"
    return panel


def main() -> None:
    """
    Build the GB panels (returns and volatility) from per-decile OOS predictions.

    Steps:
      1. Load ME1..ME10 predicted returns from results/oos_preds_gb.
      2. Load ME1..ME10 predicted volatility from results/oos_vol_gb.
      3. Stack each set into a wide month × decile matrix.
      4. Save both matrices as aligned CSVs in:
            - results/oos_panel_gb/preds_panel_gb.csv
            - results/oos_vol_panel_gb/preds_vol_panel_gb.csv
    """
    # Returns panel: uses MEj_oos_preds_gb.csv
    ret_panel = _load_pred_panel(RET_PRED_DIR, "_oos_preds_gb.csv")

    # Volatility panel: uses MEj_oos_vol_preds_gb.csv
    vol_panel = _load_pred_panel(VOL_PRED_DIR, "_oos_vol_preds_gb.csv")

    # Ensure output dirs exist
    RET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as aligned semicolon CSVs with a 'month' column
    _write_aligned_csv(ret_panel.reset_index(), RET_OUT_FILE, max_decimals=6)
    _write_aligned_csv(vol_panel.reset_index(), VOL_OUT_FILE, max_decimals=6)

    print(f"Saved GB return panel (predicted) to: {RET_OUT_FILE}")
    print(f"Saved GB vol panel   (predicted) to: {VOL_OUT_FILE}")


if __name__ == "__main__":
    main()
    