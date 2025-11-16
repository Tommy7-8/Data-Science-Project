# src/utils/build_gb_panels.py
# Collect per-decile GB OOS predictions into two panels:
#  - GB returns panel
#  - GB volatility panel
#
# Inputs (from GB training scripts):
#   results/oos_preds_gb/ME*_gb_oos_preds.csv
#   results/oos_vol_gb/ME*_gb_vol_oos_preds.csv
#
# Outputs:
#   results/gb_panels_returns/gb_oos_returns_panel.csv
#   results/gb_panels_vol/gb_oos_vol_panel.csv

from pathlib import Path
import pandas as pd

DECILES = [f"ME{i}" for i in range(1, 11)]

BASE_RESULTS = Path("results")
RET_PRED_DIR = BASE_RESULTS / "oos_preds_gb"
VOL_PRED_DIR = BASE_RESULTS / "oos_vol_gb"

RET_OUT_DIR = BASE_RESULTS / "gb_panels_returns"
VOL_OUT_DIR = BASE_RESULTS / "gb_panels_vol"

RET_OUT_FILE = RET_OUT_DIR / "gb_oos_returns_panel.csv"
VOL_OUT_FILE = VOL_OUT_DIR / "gb_oos_vol_panel.csv"


def load_panel(pred_dir: Path, suffix: str) -> pd.DataFrame:
    """Load per-decile prediction CSVs and stack into a date × decile panel."""
    series_list = []

    for dec in DECILES:
        path = pred_dir / f"{dec}{suffix}"
        if not path.exists():
            print(f"WARNING: missing {path}, skipping {dec}")
            continue

        df = pd.read_csv(path)

        if "month" not in df.columns:
            raise ValueError(f"'month' column not found in {path}")

        # 'month' should be like "YYYY-MM"
        try:
            idx = pd.PeriodIndex(df["month"].astype(str), freq="M")
        except Exception:
            idx = pd.to_datetime(df["month"], errors="coerce").dt.to_period("M")

        s = pd.Series(df["y_pred"].values, index=idx, name=dec)
        series_list.append(s)

    if not series_list:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index.name = "date"  # consistent with other monthly files
    return panel


def main():
    # returns panel
    ret_panel = load_panel(RET_PRED_DIR, "_gb_oos_preds.csv")
    # volatility panel
    vol_panel = load_panel(VOL_PRED_DIR, "_gb_vol_oos_preds.csv")

    # align on common months
    common_idx = ret_panel.index.intersection(vol_panel.index)
    ret_panel = ret_panel.loc[common_idx]
    vol_panel = vol_panel.loc[common_idx]

    RET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ret_panel.to_csv(RET_OUT_FILE)
    vol_panel.to_csv(VOL_OUT_FILE)

    print(f"Saved GB return panel to: {RET_OUT_FILE}")
    print(f"Saved GB vol panel to:    {VOL_OUT_FILE}")


if __name__ == "__main__":
    main()
