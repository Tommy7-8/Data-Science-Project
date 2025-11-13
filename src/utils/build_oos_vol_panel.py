# src/utils/build_oos_vol_panel.py
# Combine per-decile OOS VOLATILITY predictions into wide and long panels,
# and save them as visually aligned CSVs (semicolon-separated, dot decimals).
# Input: results/oos_vol_preds/ME*_oos_vol_preds.csv
# Each file has: month, y_true (true squared return), y_pred (predicted squared return)

import os
import glob
from pathlib import Path
import pandas as pd

OUT_DIR = "results/oos_vol_panel"

# ---------- pretty CSV writer (semicolon, dot decimals, aligned columns) ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    return f"{float(val):.{decimals}f}"

def _write_aligned_csv(df: pd.DataFrame, path: str, decimals: int = 6, sep: str = ";"):
    df_str = df.copy()

    # left-align only 'month'; everything else right-aligned if numeric
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-looking columns
    for c in right_cols:
        ser_num = pd.to_numeric(df_str[c], errors="coerce")
        if ser_num.notna().any():
            df_str[c] = ser_num.map(lambda v: _fmt_num(v, decimals) if pd.notna(v) else "")
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


def main():
    files = sorted(glob.glob("results/oos_vol_preds/ME*_oos_vol_preds.csv"))
    if not files:
        raise FileNotFoundError("No files found in results/oos_vol_preds/ME*_oos_vol_preds.csv")

    # Read and stack
    long_rows = []
    for p in files:
        decile = os.path.basename(p).split("_")[0]  # 'ME1' from 'ME1_oos_vol_preds.csv'
        df = pd.read_csv(p)
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
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1) Save wide panels (aligned) ---
    preds_out = preds_wide.reset_index()
    true_out  = true_wide.reset_index()
    preds_path = os.path.join(OUT_DIR, "preds_vol_panel.csv")
    true_path  = os.path.join(OUT_DIR, "true_vol_panel.csv")
    _write_aligned_csv(preds_out, preds_path, decimals=6)
    _write_aligned_csv(true_out,  true_path,  decimals=6)

    # --- 2) Coverage report ---
    cov_path = os.path.join(OUT_DIR, "coverage_vol.txt")
    with open(cov_path, "w", encoding="utf-8") as f:
        f.write(f"Months: {preds_wide.shape[0]}\n")
        f.write(f"Deciles: {preds_wide.shape[1]}\n")
        f.write(f"Start: {preds_wide.index.min()}\n")
        f.write(f"End:   {preds_wide.index.max()}\n")

    # --- 3) Console summary ---
    print("Saved VOL panels:")
    print(" -", preds_path)
    print(" -", true_path)
    print("Coverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(f" first month: {preds_wide.index.min()} | last month: {preds_wide.index.max()}")


if __name__ == "__main__":
    main()
