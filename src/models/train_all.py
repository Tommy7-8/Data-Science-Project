# src/models/train_all.py
# Train Ridge per decile and produce WALK-FORWARD predictions.

import os
import re
import glob
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


# ----------------------------- Helpers -----------------------------

def select_lagged_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
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
    return max(min_train, min(prefer_train_months, n_obs - 1))

def fit_ridge_cv(X_train, y_train, max_splits: int = 5):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(random_state=0))
    ])
    param_grid = {"ridge__alpha": [0.1, 0.3, 1.0, 3.0, 10.0]}
    n_splits = max(2, min(max_splits, max(2, X_train.shape[0] // 60)))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gscv = GridSearchCV(pipe, param_grid=param_grid, scoring="neg_mean_squared_error",
                        cv=tscv, n_jobs=-1, refit=True, verbose=0)
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_, gscv.best_params_

def unscaled_betas(model: Pipeline, feature_cols: List[str]) -> pd.DataFrame:
    scaler = model.named_steps["scaler"]
    ridge = model.named_steps["ridge"]
    coefs = ridge.coef_ / scaler.scale_
    dfb = pd.DataFrame({"feature": feature_cols, "coef": coefs})
    return dfb.sort_values("coef", ascending=False)

# -------- NEW: robust loader for aligned feature CSVs (semicolon + padded) --------
def load_feature_csv(path: str) -> pd.DataFrame:
    """
    Reads our aligned feature CSVs written by prepare_features_full.py:
      - semicolon-separated
      - cells padded for alignment
      - dot decimals
    Also works for standard (un-padded) CSVs.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", engine="python", dtype=str, encoding="utf-8-sig")
        # normalize header (strip + remove BOM)
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        # trim spaces only on string/object columns (applymap deprecated)
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())

        # convert non-date columns to numeric (dot decimals already)
        for c in df.columns:
            if c != "date":
                df[c] = pd.to_numeric(df[c].replace("", pd.NA), errors="coerce")
        return df
    else:
        # fallback for plain CSVs
        return pd.read_csv(path, encoding="utf-8-sig")

# ------------------------ Walk-forward core ------------------------

def walkforward_predictions(X: np.ndarray,
                            y: np.ndarray,
                            months: np.ndarray,
                            init_train_n: int,
                            refit_each_step: bool = True) -> Dict[str, Any]:
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

        y_hat = float(model.predict(X[t:t+1])[0])
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
        "last_params": last_params
    }

# ------------------------ Train one decile file ------------------------

def _infer_decile_from_filename(base: str) -> str:
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

def train_one_file(csv_path: str,
                   out_models: str,
                   out_preds: str,
                   out_reports: str,
                   prefer_train_months: int = 240,
                   mode: str = "walkforward") -> Dict[str, Any]:

    base = os.path.basename(csv_path)
    decile = _infer_decile_from_filename(base)

    # -------- use robust loader here --------
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
    init_train_n = first_train_size(n, prefer_train_months=prefer_train_months, min_train=120)
    if init_train_n >= n:
        raise ValueError(f"Not enough data after initial training window in {decile} (n={n}).")

    wf = walkforward_predictions(
        X, y, months,
        init_train_n=init_train_n,
        refit_each_step=(mode == "walkforward")
    )

    start_ym = df["date"].iloc[0].to_period("M")
    end_ym   = df["date"].iloc[-1].to_period("M")
    train_end_ym = df["date"].iloc[init_train_n - 1].to_period("M")
    print(f"\n=== {decile} ===")
    print(f"File: {csv_path}")
    print(f"Date range: {start_ym} → {end_ym}")
    print(f"Initial TRAIN: {start_ym} → {train_end_ym}  ({init_train_n} months)")
    print(f"OOS months: {len(wf['months'])} (through {wf['months'][-1]})")
    r2_str = "n/a" if wf["r2"] is None else f"{wf['r2']:.4f}"
    print(f"OOS R²: {r2_str} | MAE: {wf['mae']:.6f} | RMSE: {wf['rmse']:.6f}")
    if wf["last_params"] is not None:
        print(f"Last alpha: {wf['last_params']['ridge__alpha']}")

    os.makedirs(out_preds, exist_ok=True)
    preds_path = os.path.join(out_preds, f"{decile}_oos_preds.csv")
    pd.DataFrame({
        "month": wf["months"],
        "y_true": wf["y_true"],
        "y_pred": wf["y_pred"]
    }).to_csv(preds_path, index=False)

    os.makedirs(out_models, exist_ok=True)
    model_path = os.path.join(out_models, f"{decile}_ridge.pkl")
    joblib.dump({
        "model": wf["last_model"],
        "feature_cols": feature_cols,
        "init_train_n": init_train_n,
    }, model_path)

    os.makedirs(out_reports, exist_ok=True)
    report_path = os.path.join(out_reports, f"{decile}_report.txt")
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
        "model_path": model_path,
        "preds_path": preds_path,
        "report_path": report_path,
        "metrics": {"r2": wf["r2"], "mae": wf["mae"], "rmse": wf["rmse"]},
        "oos_last_month": wf["months"][-1]
    }

# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="data/processed/features_ME*_full.csv",
                        help="Glob pattern for decile feature files.")
    parser.add_argument("--models_dir", default="models",
                        help="Where to save fitted models.")
    parser.add_argument("--preds_dir", default="results/oos_preds",
                        help="Where to save OOS predictions.")
    parser.add_argument("--reports_dir", default="results/reports",
                        help="Where to save text reports.")
    parser.add_argument("--train_months", type=int, default=240,
                        help="Initial training length (months).")
    parser.add_argument("--mode", choices=["walkforward", "static"], default="walkforward",
                        help="'walkforward' (refit monthly) or 'static' (fit once).")
    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    print(f"Found {len(files)} files:")
    for f in files:
        print(" -", f)

    all_rows, errors = [], []
    for f in files:
        try:
            res = train_one_file(
                csv_path=f,
                out_models=args.models_dir,
                out_preds=args.preds_dir,
                out_reports=args.reports_dir,
                prefer_train_months=args.train_months,
                mode=args.mode
            )
            all_rows.append({
                "decile": res["decile"],
                "preds_path": res["preds_path"],
                "report_path": res["report_path"],
                "oos_r2": res["metrics"]["r2"],
                "oos_mae": res["metrics"]["mae"],
                "oos_rmse": res["metrics"]["rmse"],
                "last_oos_month": res["oos_last_month"]
            })
        except Exception as e:
            errors.append((f, str(e)))
            print(f"\n[ERROR] {f} -> {e}\n")

    os.makedirs("results", exist_ok=True)
    if all_rows:
        summary = pd.DataFrame(all_rows).sort_values("decile")
        summary.to_csv("results/train_summary.csv", index=False)
        print("\nSaved summary → results/train_summary.csv")

    if errors:
        with open("results/train_errors.txt", "w", encoding="utf-8") as f:
            for path, msg in errors:
                f.write(f"{path}: {msg}\n")
        print("Some files failed. See → results/train_errors.txt")

if __name__ == "__main__":
    main()
