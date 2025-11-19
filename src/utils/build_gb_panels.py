# src/utils/build_gb_panels.py
# Collect per-decile GB OOS predictions into two panels:
#   - GB returns panel  (predicted returns only)
#   - GB volatility panel (predicted vol only)
#
# Inputs (from GB training scripts, aligned ; CSVs):
#   results/oos_preds_gb/ME*_oos_preds_gb.csv
#   results/oos_vol_gb/ME*_oos_vol_preds_gb.csv
#
# Outputs (predicted panels only):
#   results/oos_panel_gb/preds_panel_gb.csv
#   results/oos_vol_panel_gb/preds_vol_panel_gb.csv

import os
import glob
from pathlib import Path
import pandas as pd

DECILES = [f"ME{i}" for i in range(1, 11)]

BASE_RESULTS = Path("results")
RET_PRED_DIR = BASE_RESULTS / "oos_preds_gb"
VOL_PRED_DIR = BASE_RESULTS / "oos_vol_gb"

RET_OUT_DIR = BASE_RESULTS / "oos_panel_gb"
VOL_OUT_DIR = BASE_RESULTS / "oos_vol_panel_gb"

RET_OUT_FILE = RET_OUT_DIR / "preds_panel_gb.csv"
VOL_OUT_FILE = VOL_OUT_DIR / "preds_vol_panel_gb.csv"


# ---------- robust loader for aligned OOS CSVs (semicolon + padded) ----------
def _load_oos_csv(path: Path) -> pd.DataFrame:
    """
    Load GB OOS prediction CSVs:
      - semicolon-separated, padded cells (our aligned format)
      - or plain comma CSV.
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


def _write_aligned_csv(df: pd.DataFrame, path: Path, max_decimals: int = 6, sep: str = ";"):
    df_txt = df.copy()

    # left-align only 'month'; everything else right-aligned if numeric
    left_cols = [c for c in df_txt.columns if c == "month"]
    right_cols = [c for c in df_txt.columns if c not in left_cols]

    # Format numeric-looking columns
    for c in right_cols:
        ser_num = pd.to_numeric(df_txt[c], errors="coerce")
        if ser_num.notna().any():
            df_txt[c] = ser_num.map(lambda v: _fmt_num(v, max_decimals) if pd.notna(v) else "")
        else:
            df_txt[c] = df_txt[c].astype(str)

    # Compute widths
    df_txt = df_txt.astype(str)
    widths = {c: max(len(c), df_txt[c].map(len).max()) for c in df_txt.columns}

    # Header
    header_cells = []
    for c in df_txt.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    # Rows
    lines = [sep.join(header_cells)]
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
    Load per-decile prediction CSVs (with the given suffix) and
    stack y_pred into a month × decile panel.
    """
    series_list = []

    for dec in DECILES:
        path = pred_dir / f"{dec}{suffix}"
        if not path.exists():
            print(f"WARNING: missing {path}, skipping {dec}")
            continue

        df = _load_oos_csv(path)
        if "month" not in df.columns:
            raise ValueError(f"'month' column not found in {path}")

        # Keep y_pred with month as index (string month, already YYYY-MM)
        s = pd.Series(df["y_pred"].values, index=df["month"].astype(str), name=dec)
        series_list.append(s)

    if not series_list:
        raise RuntimeError(f"No prediction files found in {pred_dir}")

    panel = pd.concat(series_list, axis=1).sort_index()
    panel.index.name = "month"
    return panel


def main():
    # Returns panel: uses MEj_oos_preds_gb.csv
    ret_panel = _load_pred_panel(RET_PRED_DIR, "_oos_preds_gb.csv")

    # Volatility panel: uses MEj_oos_vol_preds_gb.csv
    vol_panel = _load_pred_panel(VOL_PRED_DIR, "_oos_vol_preds_gb.csv")

    RET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    VOL_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save as aligned ; CSVs
    _write_aligned_csv(ret_panel.reset_index(), RET_OUT_FILE, max_decimals=6)
    _write_aligned_csv(vol_panel.reset_index(), VOL_OUT_FILE, max_decimals=6)

    print(f"Saved GB return panel (predicted) to: {RET_OUT_FILE}")
    print(f"Saved GB vol panel   (predicted) to: {VOL_OUT_FILE}")


if __name__ == "__main__":
    main()
