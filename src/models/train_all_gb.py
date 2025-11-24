# src/models/train_all_gb.py
# Train Gradient Boosting per decile and produce WALK-FORWARD *return* predictions.
#
# Outputs (per decile, GB returns):
#   - results/oos_preds_gb/MEj_oos_preds_gb.csv   (aligned ; CSV: month, y_true, y_pred)
#   - results/reports_gb/MEj_gb_walkforward_report.txt

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


# ----------------------------- Helpers -----------------------------

def select_lagged_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
    """
    Select feature columns for a given ME decile.

    We use:
      - all columns starting with f"{decile_prefix}_lag" (decile return lags)
      - all columns starting with f"{decile_prefix}_vol_" (realized vol features)
      - all columns ending with "_lag1" (factor lags: Mkt-RF_lag1, SMB_lag1, etc.)

    Args:
        df: feature DataFrame.
        decile_prefix: e.g. "ME1", "ME2", ...

    Returns:
        Ordered list of feature column names (no duplicates).
    """
    cols = df.columns.tolist()

    lag_cols = [c for c in cols if c.startswith(f"{decile_prefix}_lag")]
    vol_cols = [c for c in cols if c.startswith(f"{decile_prefix}_vol_")]
    factor_lag_cols = [c for c in cols if c.endswith("_lag1")]

    seen, ordered = set(), []
    for c in lag_cols + vol_cols + factor_lag_cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            ordered.append(c)

    return ordered


def first_train_size(n_obs: int, prefer_train_months: int = 240, min_train: int = 120) -> int:
    """
    Choose initial training window length in months.

    Rules:
      - At least min_train observations
      - At most prefer_train_months observations
      - Always leave at least 1 observation for OOS evaluation

    Args:
        n_obs: total number of observations available.
        prefer_train_months: preferred size of the initial window.
        min_train: minimal acceptable initial window size.

    Returns:
        Integer size of the first training window.
    """
    return max(min_train, min(prefer_train_months, n_obs - 1))


# -------- robust loader for aligned feature CSVs (semicolon + padded) --------

def load_feature_csv(path: str) -> pd.DataFrame:
    """
    Read the aligned feature CSVs written by prepare_features_full.py.

    These files:
      - can be semicolon-separated (aligned format) or comma-separated
      - may have padded cells for visual alignment
      - contain a 'date' column plus numeric feature columns

    Args:
        path: path to the CSV.

    Returns:
        DataFrame with:
          - 'date' as string column
          - numeric feature columns where possible.
    """
    # Peek at the first line to detect semicolon-separated aligned format
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

        # Clean header: strip whitespace and possible BOM char
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        # Strip spaces on string columns
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        # Convert non-date columns to numeric
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")

        return df
    else:
        # Fallback: plain CSV
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------- aligned CSV writer for GB OOS predictions ----------

def fmt_num(val, max_decimals: int = 6, dec_char: str = ".") -> str:
    """
    Format numbers without unnecessary trailing zeros (consistent with other scripts).

    Args:
        val: numeric value or NaN.
        max_decimals: maximum number of decimal places.
        dec_char: decimal separator ('.' by default).

    Returns:
        String representation or "" for NaN.
    """
    if pd.isna(val):
        return ""
    s = f"{float(val):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s


def write_aligned_oos_csv(
    df: pd.DataFrame,
    path: str,
    max_decimals: int = 6,
    sep: str = ";",
) -> None:
    """
    Write semicolon-separated, visually aligned CSV for OOS GB predictions.

    Assumptions:
      - First column is 'month'
      - All other columns are numeric (e.g. y_true, y_pred)

    Args:
        df: DataFrame with 'month', 'y_true', 'y_pred'.
        path: output CSV path.
        max_decimals: maximum decimals for numeric columns.
        sep: column separator (default ';').
    """
    df_txt = df.copy()

    # Format numeric columns (everything except 'month')
    for c in df_txt.columns:
        if c != "month":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v, max_decimals))

    # Convert everything to string for width calculations
    df_txt = df_txt.astype(str)

    # Compute column widths
    widths = {
        c: max(len(c), df_txt[c].map(len).max())
        for c in df_txt.columns
    }

    # Header: 'month' left-aligned, others right-aligned
    header_cells = []
    for c in df_txt.columns:
        if c == "month":
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]

    # Data rows
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
    Construct a Gradient Boosting model (HistGradientBoostingRegressor) with early stopping.

    Notes:
      - max_iter is an upper bound on the number of trees.
      - Effective number of trees is chosen via validation-based early stopping
        inside each training window.
    """
    return HistGradientBoostingRegressor(
        random_state=42,
        max_iter=1000,          # upper bound on trees
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        max_bins=64,
        early_stopping=True,
        n_iter_no_change=10,    # stop if no improvement for 10 iterations
        validation_fraction=0.2 # 20% of training window used as validation
    )


def walkforward_predictions(
    X: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    init_train_n: int,
) -> Dict[str, Any]:
    """
    Pure walk-forward scheme for GBM.

    For t = init_train_n, ..., n-1:
      - Fit a new GBM on data up to (but excluding) t, i.e. X[:t], y[:t]
      - Predict y_t on X[t]

    Args:
        X: feature matrix (n_samples, n_features).
        y: target vector (n_samples,).
        months: array of month labels aligned with y.
        init_train_n: number of initial observations for the first training window.

    Returns:
        dict with:
          - 'months': list of OOS months
          - 'y_true': np.array of OOS realized returns
          - 'y_pred': np.array of OOS predicted returns
          - 'rmse', 'mae', 'r2'
    """
    n = len(y)
    preds, truths, pred_months = [], [], []

    for t in range(init_train_n, n):
        model = make_gbm()
        # Expanding window: use all data up to t
        model.fit(X[:t], y[:t])

        y_hat = float(model.predict(X[t:t + 1])[0])
        preds.append(y_hat)
        truths.append(float(y[t]))
        pred_months.append(str(months[t]))

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
    }


# ------------------------ Train one decile file ------------------------

def _infer_decile_from_filename(base: str) -> str:
    """
    Extract the decile label (e.g. "ME1") from a feature filename.

    We attempt:
      - pattern like "features_ME1_full.csv"
      - pattern like "features_ME1.csv"
      - any underscore-separated token that starts with "ME" and ends with digits

    Raises:
        ValueError if no decile can be inferred.
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


def train_one_file_gb(
    csv_path: str,
    out_preds: str,
    out_reports: str,
    prefer_train_months: int = 240,
) -> Dict[str, Any]:
    """
    Train a GBM for one ME decile and produce walk-forward OOS return predictions.

    Steps:
      - infer decile label from filename
      - load features
      - define next-month return as target
      - build lag/vol/factor-lag feature set
      - pick initial training window
      - run GBM walk-forward predictions with early stopping
      - save aligned OOS prediction CSV and text report

    Args:
        csv_path: path to features_ME*_full.csv.
        out_preds: directory to save OOS prediction CSVs.
        out_reports: directory to save text reports.
        prefer_train_months: preferred initial training window size.

    Returns:
        dict with metadata (paths, metrics, last month, etc.).
    """
    base = os.path.basename(csv_path)
    decile = _infer_decile_from_filename(base)

    df = load_feature_csv(csv_path)

    # Basic time ordering and checks
    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if decile not in df.columns:
        raise ValueError(f"{decile} column not found in {csv_path}")

    # --- Define RETURN target: next-month return ---
    df["target"] = df[decile].shift(-1)
    df["target_month"] = df["date"].shift(-1).dt.to_period("M").astype(str)
    # Drop rows without a valid target (typically the last row)
    df = df.dropna(subset=["target", "target_month"]).copy()

    # Feature set: decile lags, vol features, factor lags
    feature_cols = select_lagged_features(df, decile_prefix=decile)
    if len(feature_cols) == 0:
        raise ValueError(f"No lagged features found in {csv_path} for {decile}.")

    X = df[feature_cols].values
    y = df["target"].values
    months = df["target_month"].values

    # Initial training window
    n = len(df)
    init_train_n = first_train_size(
        n,
        prefer_train_months=prefer_train_months,
        min_train=120,
    )
    if init_train_n >= n:
        raise ValueError(
            f"Not enough data after initial training window in {decile} (n={n})."
        )

    # Walk-forward GB predictions
    wf = walkforward_predictions(
        X=X,
        y=y,
        months=months,
        init_train_n=init_train_n,
    )

    start_ym = df["date"].iloc[0].to_period("M")
    end_ym = df["date"].iloc[-1].to_period("M")
    train_end_ym = df["date"].iloc[init_train_n - 1].to_period("M")

    # Console summary
    print(f"\n=== {decile} (GB RETURN, walkforward with early stopping) ===")
    print(f"File: {csv_path}")
    print(f"Date range: {start_ym} → {end_ym}")
    print(f"Initial TRAIN: {start_ym} → {train_end_ym}  ({init_train_n} months)")
    print(f"OOS months: {len(wf['months'])} (through {wf['months'][-1]})")
    r2_str = "n/a" if wf["r2"] is None else f"{wf['r2']:.4f}"
    print(
        f"OOS R² (return): {r2_str} | "
        f"MAE: {wf['mae']:.6f} | RMSE: {wf['rmse']:.6f}"
    )

    # --- Save predictions (aligned ; CSV) ---
    os.makedirs(out_preds, exist_ok=True)
    preds_path = os.path.join(out_preds, f"{decile}_oos_preds_gb.csv")

    preds_df = pd.DataFrame({
        "month": wf["months"],
        "y_true": wf["y_true"],   # realized next-month return
        "y_pred": wf["y_pred"],   # predicted next-month return
    })
    write_aligned_oos_csv(preds_df, preds_path, max_decimals=6)

    # --- Save text report ---
    os.makedirs(out_reports, exist_ok=True)
    report_path = os.path.join(out_reports, f"{decile}_gb_walkforward_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"decile: {decile}\n")
        f.write(f"file: {csv_path}\n")
        f.write(f"date_range: {start_ym} → {end_ym}\n")
        f.write(f"initial_train_months: {init_train_n} (through {train_end_ym})\n")
        f.write("mode: walkforward_early_stopping\n")
        f.write(f"oos_months: {len(wf['months'])} (through {wf['months'][-1]})\n")
        f.write(f"oos_r2: {wf['r2']}\n")
        f.write(f"oos_mae: {wf['mae']}\n")
        f.write(f"oos_rmse: {wf['rmse']}\n")

    return {
        "decile": decile,
        "preds_path": preds_path,
        "report_path": report_path,
        "metrics": {
            "r2": wf["r2"],
            "mae": wf["mae"],
            "rmse": wf["rmse"],
        },
        "oos_last_month": wf["months"][-1],
    }


# ----------------------------- Main -----------------------------

def main():
    """
    CLI entry point: train GBM return models for all ME deciles.

    It:
      - finds all features_ME*_full.csv files
      - trains a GBM per decile using walk-forward + early stopping
      - saves OOS prediction CSVs and text reports
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="data/processed/features_ME*_full.csv",
        help="Glob pattern for decile feature files.",
    )
    parser.add_argument(
        "--preds_dir",
        default="results/oos_preds_gb",
        help="Where to save OOS GB predictions.",
    )
    parser.add_argument(
        "--reports_dir",
        default="results/reports_gb",
        help="Where to save GB text reports.",
    )
    parser.add_argument(
        "--train_months",
        type=int,
        default=240,
        help="Initial training length (months).",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    print("Found {} feature files for GB RETURN training (walkforward + early stopping):"
          .format(len(files)))
    for fpath in files:
        print(" -", fpath)

    # Train GBM for each decile feature file;
    # keep going even if one file fails.
    for fpath in files:
        try:
            _ = train_one_file_gb(
                csv_path=fpath,
                out_preds=args.preds_dir,
                out_reports=args.reports_dir,
                prefer_train_months=args.train_months,
            )
        except Exception as e:
            print(f"\n[ERROR] {fpath} -> {e}\n")


if __name__ == "__main__":
    main()
