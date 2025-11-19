# src/utils/build_lr_panel.py
# Combine per-decile OOS *LR* predictions into wide panels for:
#   - RETURNS  (results/oos_preds_lr/ME*_oos_preds_lr.csv)
#   - VOLATILITY (results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv)
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

RET_OUT_DIR = "results/oos_panel_lr"
VOL_OUT_DIR = "results/oos_vol_panel_lr"

# ---------- robust loader for aligned OOS CSVs (semicolon + padded) ----------
def _load_oos_csv(path: str) -> pd.DataFrame:
    """
    Load aligned OOS prediction CSVs written by train_all_lr.py / train_all_vol_lr.py:
      - semicolon-separated
      - cells padded for alignment
      - dot decimals
    Also works for plain CSVs.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", engine="python", dtype=str, encoding="utf-8-sig")
        # normalize header
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        # strip spaces on string columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        # convert numeric cols (except 'month')
        for c in df.columns:
            if c != "month":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")
        return df
    else:
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------- pretty CSV writer (semicolon, aligned columns, trimmed zeros) ----------
def _fmt_num(val, max_decimals=6, dec_char="."):
    if pd.isna(val):
        return ""
    s = f"{float(val):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s

def _write_aligned_csv(df: pd.DataFrame, path: str, max_decimals: int = 6, sep: str = ";"):
    df_str = df.copy()

    # left-align only 'month'; everything else right-aligned if numeric
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-looking columns
    for c in right_cols:
        ser_num = pd.to_numeric(df_str[c], errors="coerce")
        if ser_num.notna().any():
            df_str[c] = ser_num.map(lambda v: _fmt_num(v, max_decimals) if pd.notna(v) else "")
        else:
            df_str[c] = df_str[c].astype(str)

    # Compute widths
    df_str = df_str.astype(str)
    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    # Header
    header_cells = []
    for c in df_str.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    # Rows
    lines = [sep.join(header_cells)]
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
def build_return_panels():
    files = sorted(glob.glob("results/oos_preds_lr/ME*_oos_preds_lr.csv"))
    if not files:
        raise FileNotFoundError("No files found in results/oos_preds_lr/ME*_oos_preds_lr.csv")

    # Read and stack
    long_rows = []
    for p in files:
        decile = os.path.basename(p).split("_")[0]  # 'ME1' from 'ME1_oos_preds_lr.csv'
        df = _load_oos_csv(p)
        need = {"month", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        df = df[["month", "y_true", "y_pred"]].copy()
        df["decile"] = decile
        long_rows.append(df)

    long = pd.concat(long_rows, axis=0, ignore_index=True)

    # Enforce ME1 → ME10 order inside each month
    long["dec_num"] = long["decile"].str.extract(r"ME(\d+)").astype(int)
    long = long.sort_values(["month", "dec_num"]).drop(columns="dec_num").reset_index(drop=True)

    # Pivot to wide (preds and trues), months ascending
    preds_wide = long.pivot(index="month", columns="decile", values="y_pred").sort_index()
    true_wide  = long.pivot(index="month", columns="decile", values="y_true").sort_index()

    # Ensure deciles are ordered ME1..ME10
    ordered_cols = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in preds_wide.columns]
    preds_wide = preds_wide[ordered_cols]
    true_wide  = true_wide[ordered_cols]

    # Create output dir
    os.makedirs(RET_OUT_DIR, exist_ok=True)

    # --- Save wide panels (aligned) ---
    preds_out = preds_wide.reset_index()
    true_out  = true_wide.reset_index()
    preds_path = os.path.join(RET_OUT_DIR, "preds_panel_lr.csv")
    true_path  = os.path.join(RET_OUT_DIR, "true_panel_lr.csv")
    _write_aligned_csv(preds_out, preds_path, max_decimals=6)
    _write_aligned_csv(true_out,  true_path,  max_decimals=6)

    # Console summary
    print("Saved LR RETURN panels:")
    print(" -", preds_path)
    print(" -", true_path)
    print("Coverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(f" first month: {preds_wide.index.min()} | last month: {preds_wide.index.max()}")


# ---------- VOLATILITY PANELS (from results/oos_vol_preds_lr) ----------
def build_vol_panels():
    files = sorted(glob.glob("results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv"))
    if not files:
        raise FileNotFoundError("No files found in results/oos_vol_preds_lr/ME*_oos_vol_preds_lr.csv")

    # Read and stack
    long_rows = []
    for p in files:
        decile = os.path.basename(p).split("_")[0]  # 'ME1' from 'ME1_oos_vol_preds_lr.csv'
        df = _load_oos_csv(p)
        need = {"month", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        df = df[["month", "y_true", "y_pred"]].copy()
        df["decile"] = decile
        long_rows.append(df)

    long = pd.concat(long_rows, axis=0, ignore_index=True)

    # Enforce ME1 → ME10 order inside each month
    long["dec_num"] = long["decile"].str.extract(r"ME(\d+)").astype(int)
    long = long.sort_values(["month", "dec_num"]).drop(columns="dec_num").reset_index(drop=True)

    # Pivot to wide (preds and trues), months ascending
    preds_wide = long.pivot(index="month", columns="decile", values="y_pred").sort_index()
    true_wide  = long.pivot(index="month", columns="decile", values="y_true").sort_index()

    # Ensure deciles are ordered ME1..ME10
    ordered_cols = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in preds_wide.columns]
    preds_wide = preds_wide[ordered_cols]
    true_wide  = true_wide[ordered_cols]

    # Create output dir
    os.makedirs(VOL_OUT_DIR, exist_ok=True)

    # --- Save predicted and true VOL panels (aligned) ---
    preds_out = preds_wide.reset_index()
    true_out  = true_wide.reset_index()
    preds_path = os.path.join(VOL_OUT_DIR, "preds_vol_panel_lr.csv")
    true_path  = os.path.join(VOL_OUT_DIR, "true_vol_panel_lr.csv")
    _write_aligned_csv(preds_out, preds_path, max_decimals=6)
    _write_aligned_csv(true_out,  true_path,  max_decimals=6)

    # Console summary
    print("\nSaved LR VOL panels:")
    print(" -", preds_path)
    print(" -", true_path)
    print("Coverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(f" first month: {preds_wide.index.min()} | last month: {preds_wide.index.max()}")


def main():
    # Build both LR return and LR volatility panels
    build_return_panels()
    build_vol_panels()


if __name__ == "__main__":
    main()
