# src/models/train_all_gb_vol.py
# Train Gradient Boosting per decile and produce WALK-FORWARD *volatility* predictions (GB version).
#
# Outputs (per decile, GB VOL):
#   - results/oos_vol_gb/MEj_oos_vol_preds_gb.csv    (aligned ; CSV: month, y_true, y_pred)
#   - results/reports_gb_vol/MEj_gb_vol_<mode>_report.txt

import os
import re
import glob
import argparse
from typing import List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

N_LAGS = 12  # kept for reference, not directly used


# ----------------------------- Helpers -----------------------------

def select_vol_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
    """
    Features for VOLATILITY model:
      - decile's own *return* lags (MEk_lag1..)
      - decile's own vol features / vol lags (MEk_vol_*)
      - factor lags (e.g., Mkt_RF_lag1, SMB_lag1, etc.)
    """
    cols = df.columns.tolist()

    ret_lag_cols = [c for c in cols if c.startswith(f"{decile_prefix}_lag")]
    vol_cols = [c for c in cols if c.startswith(f"{decile_prefix}_vol_")]
    factor_lag_cols = [c for c in cols if c.endswith("_lag1")]

    seen, ordered = set(), []
    for c in ret_lag_cols + vol_cols + factor_lag_cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


def infer_vol_target_col(df: pd.DataFrame, decile_prefix: str) -> str:
    """
    Try to infer the 'next-month vol' base column for this decile.
    We assume columns like:
      - ME1_vol_1, ME1_vol_3, ME1_vol_6, ...
    and we want the *shortest* horizon (e.g. vol_1).
    """
    vol_candidates = [c for c in df.columns if c.startswith(f"{decile_prefix}_vol")]
    if not vol_candidates:
        raise ValueError(f"No volatility columns found for {decile_prefix} in dataframe.")

    # Prefer explicit _vol_1 or vol1 if present
    for pat in (f"{decile_prefix}_vol_1", f"{decile_prefix}_vol1"):
        if pat in vol_candidates:
            return pat

    # Otherwise pick the one with smallest numeric suffix after 'vol'
    def _horizon(col: str) -> int:
        m = re.search(r"vol[_]?(\d+)", col)
        return int(m.group(1)) if m else 9999

    vol_candidates.sort(key=_horizon)
    return vol_candidates[0]


def first_train_size(n_obs: int, prefer_train_months: int = 240, min_train: int = 120) -> int:
    return max(min_train, min(prefer_train_months, n_obs - 1))


# -------- robust loader for aligned feature CSVs (semicolon + padded) --------

def load_feature_csv(path: str) -> pd.DataFrame:
    """
    Reads the aligned feature CSVs written by prepare_features_full.py:
      - semicolon-separated OR comma-separated
      - cells may be padded with spaces
      - 'date' column as string, other columns numeric
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", engine="python", dtype=str, encoding="utf-8-sig")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")
        return df
    else:
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------- aligned CSV writer for GB VOL OOS predictions ----------

def fmt_num(val, max_decimals: int = 6, dec_char: str = ".") -> str:
    """Format numbers without unnecessary trailing zeros."""
    if pd.isna(val):
        return ""
    s = f"{float(val):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s


def write_aligned_oos_csv(df: pd.DataFrame, path: str, max_decimals: int = 6, sep: str = ";"):
    """
    Writes semicolon-separated, visually aligned CSV for OOS GB VOL predictions.
    Assumes first column is 'month' and others are numeric.
    """
    df_txt = df.copy()
    for c in df_txt.columns:
        if c != "month":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v, max_decimals))

    df_txt = df_txt.astype(str)
    widths = {c: max(len(c), df_txt[c].map(len).max()) for c in df_txt.columns}

    header_cells = []
    for c in df_txt.columns:
        if c == "month":
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]
    for _, row in df_txt.iterrows():
        cells = []
        for c in df_txt.columns:
            val = row[c]
            if c == "month":
                cells.append(f"{val:<{widths[c]}}")
            else:
                cells.append(f"{val:>{widths[c]}}")
        lines.append(sep.join(cells))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ------------------------ GBM training & prediction ------------------------

def make_gbm() -> HistGradientBoostingRegressor:
    """
    Fixed, reasonably good and fast GBM config.
    """
    return HistGradientBoostingRegressor(
        random_state=0,
        max_iter=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        max_bins=64,
    )


def walkforward_predictions(
    X: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    init_train_n: int,
    mode: str = "static",
) -> Dict[str, Any]:
    """
    Two modes:

      - static: fit once on first `init_train_n` months, predict all later months.
      - walkforward: refit GBM each month on an expanding window (slower, more adaptive).
    """
    n = len(y)
    preds, truths, pred_months = [], [], []
    model = None

    if mode == "static":
        model = make_gbm()
        model.fit(X[:init_train_n], y[:init_train_n])
        for t in range(init_train_n, n):
            y_hat = float(model.predict(X[t:t+1])[0])
            preds.append(y_hat)
            truths.append(float(y[t]))
            pred_months.append(str(months[t]))
    elif mode == "walkforward":
        for t in range(init_train_n, n):
            model = make_gbm()
            model.fit(X[:t], y[:t])
            y_hat = float(model.predict(X[t:t+1])[0])
            preds.append(y_hat)
            truths.append(float(y[t]))
            pred_months.append(str(months[t]))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    y_pred = np.array(preds)
    y_true = np.array(truths)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan")

    return {
        "months": pred_months,
        "y_true": y_true,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "r2": None if np.isnan(r2) else r2,
        "model": model,
    }


# ------------------------ Train one decile file ------------------------

def _infer_decile_from_filename(base: str) -> str:
    """
    Extract something like 'ME1' from paths such as:
      - .../features_ME1_full.csv
      - .../my_features_ME3.csv
    """
    m = re.search(r"features_(ME\d+)_", base)
    if m:
        return m.group(1)
    m2 = re.search(r"features_(ME\d+)\.csv$", base)
    if m2:
        return m2.group(1)
    for p in base.split("_"):
        if p.startswith("ME") and p[2:].isdigit():
            return p
    raise ValueError(f"Could not infer decile from filename: {base}")


def train_one_file_gb_vol(
    csv_path: str,
    out_preds: str,
    out_reports: str,
    prefer_train_months: int = 240,
    mode: str = "static",
) -> Dict[str, Any]:
    base = os.path.basename(csv_path)
    decile = _infer_decile_from_filename(base)

    df = load_feature_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Infer volatility target column (e.g. ME1_vol_1) ---
    vol_target_col = infer_vol_target_col(df, decile)

    # Next-month volatility as target
    df["target"] = df[vol_target_col].shift(-1)
    df["target_month"] = df["date"].shift(-1).dt.to_period("M").astype(str)
    df = df.dropna(subset=["target", "target_month"]).copy()

    feature_cols = select_vol_features(df, decile_prefix=decile)
    if len(feature_cols) == 0:
        raise ValueError(f"No volatility feature columns found for {decile} in {csv_path}.")

    X = df[feature_cols].values
    y = df["target"].values
    months = df["target_month"].values

    n = len(df)
    init_train_n = first_train_size(n, prefer_train_months=prefer_train_months, min_train=120)
    if init_train_n >= n:
        raise ValueError(f"Not enough data after initial training window in {decile} (n={n}).")

    wf = walkforward_predictions(
        X=X,
        y=y,
        months=months,
        init_train_n=init_train_n,
        mode=mode,
    )

    start_ym = df["date"].iloc[0].to_period("M")
    end_ym = df["date"].iloc[-1].to_period("M")
    train_end_ym = df["date"].iloc[init_train_n - 1].to_period("M")

    print(f"\n=== {decile} (GB VOL, mode={mode}) ===")
    print(f"File: {csv_path}")
    print(f"Date range: {start_ym} → {end_ym}")
    print(f"Initial TRAIN: {start_ym} → {train_end_ym}  ({init_train_n} months)")
    print(f"OOS months: {len(wf['months'])} (through {wf['months'][-1]})")
    r2_str = "n/a" if wf['r2'] is None else f"{wf['r2']:.4f}"
    print(f"OOS R² (vol): {r2_str} | MAE: {wf['mae']:.6f} | RMSE: {wf['rmse']:.6f}")
    print(f"Vol target column: {vol_target_col}")

    # --- Save VOL predictions (aligned ; CSV) ---
    os.makedirs(out_preds, exist_ok=True)
    preds_path = os.path.join(out_preds, f"{decile}_oos_vol_preds_gb.csv")
    preds_df = pd.DataFrame(
        {
            "month": wf["months"],
            "y_true": wf["y_true"],  # true next-month vol
            "y_pred": wf["y_pred"],  # predicted next-month vol
        }
    )
    write_aligned_oos_csv(preds_df, preds_path, max_decimals=6)

    # --- Save VOL report ---
    os.makedirs(out_reports, exist_ok=True)
    report_path = os.path.join(out_reports, f"{decile}_gb_vol_{mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"decile: {decile}\n")
        f.write(f"file: {csv_path}\n")
        f.write(f"date_range: {start_ym} → {end_ym}\n")
        f.write(f"initial_train_months: {init_train_n} (through {train_end_ym})\n")
        f.write(f"mode: {mode}\n")
        f.write(f"vol_target_col: {vol_target_col}\n")
        f.write(f"oos_months: {len(wf['months'])} (through {wf['months'][-1]})\n")
        f.write(f"oos_r2: {wf['r2']}\n")
        f.write(f"oos_mae: {wf['mae']}\n")
        f.write(f"oos_rmse: {wf['rmse']}\n")

    return {
        "decile": decile,
        "preds_path": preds_path,
        "report_path": report_path,
        "metrics": {"r2": wf["r2"], "mae": wf["mae"], "rmse": wf["rmse"]},
        "oos_last_month": wf["months"][-1],
    }


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="data/processed/features_ME*_full.csv",
        help="Glob pattern for decile feature files.",
    )
    parser.add_argument(
        "--preds_dir",
        default="results/oos_vol_gb",
        help="Where to save OOS GB VOL predictions.",
    )
    parser.add_argument(
        "--reports_dir",
        default="results/reports_gb_vol",
        help="Where to save GB VOL text reports.",
    )
    parser.add_argument(
        "--train_months",
        type=int,
        default=240,
        help="Initial training length (months).",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "walkforward"],
        default="static",  # fast by default
        help="static = fit once; walkforward = refit each month (slower).",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    print(f"Found {len(files)} feature files for GB VOL training:")
    for fpath in files:
        print(" -", fpath)

    for fpath in files:
        try:
            _ = train_one_file_gb_vol(
                csv_path=fpath,
                out_preds=args.preds_dir,
                out_reports=args.reports_dir,
                prefer_train_months=args.train_months,
                mode=args.mode,
            )
        except Exception as e:
            print(f"\n[ERROR] {fpath} -> {e}\n")


if __name__ == "__main__":
    main()
