# src/models/train_all_lr.py
# Train Ridge per decile and produce WALK-FORWARD predictions (LR version).

import os
import re
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ----------------------------- Helpers -----------------------------

def select_lagged_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
    """
    Select feature columns for a given decile.

    We include:
      - all columns starting with f"{decile_prefix}_lag" (e.g. ME1_lag1..)
      - all columns starting with f"{decile_prefix}_vol_" (e.g. ME1_vol_12m)
      - all columns ending with "_lag1" (factor lags like Mkt-RF_lag1, SMB_lag1)

    The order preserves: lags, then vols, then factor lags; duplicates are removed.
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
    Decide how many initial observations to use for the first training window.

    Logic:
      - never use fewer than min_train
      - try to use prefer_train_months (e.g. 240 months = 20 years)
      - but leave at least 1 observation for OOS prediction
    """
    return max(min_train, min(prefer_train_months, n_obs - 1))


def fit_ridge_cv(X_train, y_train, max_splits: int = 5):
    """
    Fit a Ridge regression with standardization and time-series cross-validation.

    Steps:
      - Build a Pipeline: StandardScaler -> Ridge
      - Set up a small grid for alpha
      - Use TimeSeriesSplit with a number of splits depending on sample size
      - Optimize using negative MSE

    Returns:
        (best_estimator, best_params_dict)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(random_state=0)),
    ])

    param_grid = {"ridge__alpha": [0.1, 0.3, 1.0, 3.0, 10.0]}

    n_splits = max(2, min(max_splits, max(2, X_train.shape[0] // 60)))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gscv.fit(X_train, y_train)

    return gscv.best_estimator_, gscv.best_params_


def unscaled_betas(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    """
    Recover coefficients in the original (unscaled) feature space.

    The Ridge is fit on standardized features, so we divide by the scaler's
    standard deviations to get back to the original scale.
    """
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]

    coefs = ridge.coef_ / scaler.scale_
    dfb = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    return dfb.sort_values("coef", ascending=False)


# -------- robust loader for aligned feature CSVs (semicolon + padded) --------

def load_feature_csv(path: str) -> pd.DataFrame:
    """
    Read our aligned feature CSVs written by prepare_features_full.py.

    These files:
      - are semicolon-separated
      - may have padded cells for alignment
      - use dot decimals
      - have 'date' as the time column

    This loader also works for standard, non-aligned CSVs.
    """
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

        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")

        return df
    else:
        return pd.read_csv(path, encoding="utf-8-sig")


# ---------- aligned CSV writer for LR OOS predictions ----------

def fmt_num(val, max_decimals=6, dec_char="."):
    """
    Format numbers without unnecessary trailing zeros (consistent with other scripts).
    """
    if pd.isna(val):
        return ""
    s = f"{float(val):.{max_decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if dec_char != ".":
        s = s.replace(".", dec_char)
    return s


def write_aligned_oos_csv(df: pd.DataFrame, path: str, max_decimals: int = 6):
    """
    Write semicolon-separated, visually aligned CSV for OOS LR predictions.

    Assumptions:
        - First column is 'month' (string YYYY-MM).
        - All other columns are numeric.
    """
    df_txt = df.copy()

    for c in df_txt.columns:
        if c != "month":
            df_txt[c] = df_txt[c].map(lambda v: fmt_num(v, max_decimals))

    widths = {
        c: max(len(c), df_txt[c].astype(str).map(len).max())
        for c in df_txt.columns
    }

    header_cells = [f"{'month':<{widths['month']}}"] + [
        f"{c:>{widths[c]}}" for c in df_txt.columns if c != "month"
    ]

    lines = [";".join(header_cells)]

    for _, row in df_txt.iterrows():
        left = f"{str(row['month']):<{widths['month']}}"
        nums = [f"{str(row[c]):>{widths[c]}}" for c in df_txt.columns if c != "month"]
        lines.append(";".join([left] + nums))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ------------------------ Walk-forward core ------------------------

def walkforward_predictions(
    X: np.ndarray,
    y: np.ndarray,
    months: np.ndarray,
    init_train_n: int,
    refit_each_step: bool = True,
) -> Dict[str, Any]:
    """
    Perform walk-forward (or static) predictions with Ridge.

    Modes:
      - refit_each_step=True ("walkforward"):
            re-fit Ridge with CV every month using all data up to t
      - refit_each_step=False ("static"):
            fit once on the initial training window, then keep using it
    """
    n = len(y)
    preds, truths, pred_months = [], [], []
    last_model, last_params = None, None

    if not refit_each_step:
        model, params = fit_ridge_cv(X[:init_train_n], y[:init_train_n])
        last_model, last_params = model, params

    for t in range(init_train_n, n):
        if refit_each_step:
            model, params = fit_ridge_cv(X[:t], y[:t])
            last_model, last_params = model, params
        else:
            model, params = last_model, last_params

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
        "last_model": last_model,
        "last_params": last_params,
    }


# ------------------------ Train one decile file ------------------------

def _infer_decile_from_filename(base: str) -> str:
    """
    Infer the decile label (e.g. 'ME1') from a feature filename.

    We try a few patterns:
      - 'features_(ME\\d+)_...'
      - 'features_(ME\\d+).csv'
      - any token starting with 'ME' followed by digits

    Raises:
        ValueError if nothing matches.
    """
    m = re.search(r"features_(ME\d+)_", base)
    if m:
        return m.group(1)

    m2 = re.search(r"features_(ME\d+)\.csv$", base)
    if m2:
        return m2.group(1)

    parts = base.split("_")
    for p in parts:
        if p.startswith("ME") and p[2:].isdigit():
            return p

    raise ValueError(f"Could not infer decile from filename: {base}")


def train_one_file(
    csv_path: str,
    out_preds: str,
    out_reports: str,
    prefer_train_months: int = 240,
    mode: str = "walkforward",
) -> Dict[str, Any]:
    """
    Train Ridge for a single decile feature file and produce OOS predictions.
    """
    base = os.path.basename(csv_path)
    decile = _infer_decile_from_filename(base)

    df = load_feature_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if decile not in df.columns:
        raise ValueError(f"{decile} column not found in {csv_path}")

    df["target"] = df[decile].shift(-1)
    df["target_month"] = df["date"].shift(-1).dt.to_period("M").astype(str)
    df = df.dropna(subset=["target", "target_month"]).copy()

    feature_cols = select_lagged_features(df, decile_prefix=decile)
    if len(feature_cols) == 0:
        raise ValueError(f"No lagged features found in {csv_path} for {decile}.")

    X = df[feature_cols].values
    y = df["target"].values
    months = df["target_month"].values

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

    wf = walkforward_predictions(
        X,
        y,
        months,
        init_train_n=init_train_n,
        refit_each_step=(mode == "walkforward"),
    )

    start_ym = df["date"].iloc[0].to_period("M")
    end_ym = df["date"].iloc[-1].to_period("M")
    train_end_ym = df["date"].iloc[init_train_n - 1].to_period("M")

    print(f"\n=== {decile} (LR) ===")
    print(f"File: {csv_path}")
    print(f"Date range: {start_ym} → {end_ym}")
    print(f"Initial TRAIN: {start_ym} → {train_end_ym}  ({init_train_n} months)")
    print(f"OOS months: {len(wf['months'])} (through {wf['months'][-1]})")

    r2_str = "n/a" if wf["r2"] is None else f"{wf['r2']:.4f}"
    print(f"OOS R²: {r2_str} | MAE: {wf['mae']:.6f} | RMSE: {wf['rmse']:.6f}")
    if wf["last_params"] is not None:
        print(f"Last alpha: {wf["last_params"]["ridge__alpha"]}")

    # --- Save OOS predictions (LR) as aligned CSV ---
    os.makedirs(out_preds, exist_ok=True)
    preds_path = os.path.join(out_preds, f"{decile}_oos_preds_lr.csv")

    preds_df = pd.DataFrame({
        "month": wf["months"],
        "y_true": wf["y_true"],
        "y_pred": wf["y_pred"],
    })
    write_aligned_oos_csv(preds_df, preds_path, max_decimals=6)

    # --- Save text report for debugging/inspection (LR) ---
    os.makedirs(out_reports, exist_ok=True)
    report_path = os.path.join(out_reports, f"{decile}_report_lr.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"decile: {decile}\n")
        f.write(f"file: {csv_path}\n")
        f.write(f"date_range: {start_ym} → {end_ym}\n")
        f.write(f"initial_train_months: {init_train_n} (through {train_end_ym})\n")
        f.write(f"oos_months: {len(wf['months'])} (through {wf['months'][-1]})\n")
        f.write(f"oos_r2: {wf['r2']}\n")
        f.write(f"oos_mae: {wf['mae']}\n")
        f.write(f"oos_rmse: {wf['rmse']}\n")
        if wf["last_params"] is not None:
            f.write(f"last_alpha: {wf['last_params']['ridge__alpha']}\n")

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
    CLI: train LR models for all deciles based on feature CSVs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glob",
        default="data/processed/features_ME*_full.csv",
        help="Glob pattern for decile feature files.",
    )
    parser.add_argument(
        "--preds_dir",
        default="results/oos_preds_lr",
        help="Where to save LR OOS predictions.",
    )
    parser.add_argument(
        "--reports_dir",
        default="results/reports_lr",
        help="Where to save LR text reports.",
    )
    parser.add_argument(
        "--train_months",
        type=int,
        default=240,
        help="Initial training length (months).",
    )
    parser.add_argument(
        "--mode",
        choices=["walkforward", "static"],
        default="walkforward",
        help="'walkforward' (refit monthly) or 'static' (fit once).",
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    print(f"Found {len(files)} feature files for LR:")
    for f in files:
        print(" -", f)

    for fpath in files:
        try:
            train_one_file(
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
