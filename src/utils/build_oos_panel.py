# src/utils/build_oos_panel.py
# Combine per-decile OOS predictions into wide and long panels.

import os
import glob
import pandas as pd

OUT_DIR = "results/oos_panel"

def main():
    files = sorted(glob.glob("results/oos_preds/ME*_oos_preds.csv"))
    if not files:
        raise FileNotFoundError("No files found in results/oos_preds/ME*_oos_preds.csv")

    # Read and stack in long format
    long_rows = []
    for p in files:
        decile = os.path.basename(p).split("_")[0]  # 'ME1' from 'ME1_oos_preds.csv'
        df = pd.read_csv(p)
        # Basic checks
        need = {"month", "y_true", "y_pred"}
        if not need.issubset(df.columns):
            raise ValueError(f"{p} missing columns {need - set(df.columns)}")
        # Keep only necessary cols
        df = df[["month", "y_true", "y_pred"]].copy()
        df["decile"] = decile
        long_rows.append(df)

    long = pd.concat(long_rows, axis=0, ignore_index=True)
    long = long.sort_values(["month", "decile"]).reset_index(drop=True)

    # Pivot to wide (preds and trues)
    preds_wide = long.pivot(index="month", columns="decile", values="y_pred").sort_index()
    true_wide  = long.pivot(index="month", columns="decile", values="y_true").sort_index()

    # Ensure deciles are ordered ME1..ME10
    ordered_cols = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in preds_wide.columns]
    preds_wide = preds_wide[ordered_cols]
    true_wide  = true_wide[ordered_cols]

    # Create output dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save wide panels with 'month' as first column
    preds_out = preds_wide.reset_index()
    true_out  = true_wide.reset_index()
    preds_out.to_csv(os.path.join(OUT_DIR, "preds_panel.csv"), index=False)
    true_out.to_csv(os.path.join(OUT_DIR, "true_panel.csv"), index=False)

    # Save long panel
    long.to_csv(os.path.join(OUT_DIR, "combined_long.csv"), index=False)

    # Small coverage report
    with open(os.path.join(OUT_DIR, "coverage.txt"), "w", encoding="utf-8") as f:
        f.write(f"Months: {preds_wide.shape[0]}\n")
        f.write(f"Deciles: {preds_wide.shape[1]}\n")
        f.write(f"Start: {preds_wide.index.min()}\n")
        f.write(f"End:   {preds_wide.index.max()}\n")

    # Console summary
    print("Saved:")
    print(" -", os.path.join(OUT_DIR, "preds_panel.csv"))
    print(" -", os.path.join(OUT_DIR, "true_panel.csv"))
    print(" -", os.path.join(OUT_DIR, "combined_long.csv"))
    print("\nCoverage:")
    print(f" rows (months): {preds_wide.shape[0]}")
    print(f" cols (deciles): {preds_wide.shape[1]}")
    print(f" first month: {preds_wide.index.min()} | last month: {preds_wide.index.max()}")

if __name__ == "__main__":
    main()