# src/models/train_all_gb.py
# Train Gradient Boosting per decile and produce WALK-FORWARD *return* predictions
# (fast version: one GBM per decile, no per-step GridSearchCV)

import os
import re
import glob
import argparse
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib


# ----------------------------- Helpers -----------------------------


def select_lagged_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
    """
    Use decile's own lags + its realized vol features + factor lags.
    Assumes columns like:
      - ME1_lag1, ME1_lag2, ...
      - ME1_vol_1, ME1_vol_3, ...
      - Mkt_RF_lag1, SMB_lag1, HML_lag1, etc. (whatever your pipeline created)
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
    """
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


# ------------------------ GBM training & prediction ------------------------


def make_gbm() -> HistGradientBoostingRegressor:
    """
    Fixed, reasonably fast GBM config (no grid search).
    You can tweak these hyperparameters if needed.
    """
    return HistGradientBoostingRegressor(
        random_state=0,
        max_iter=300,        # total trees
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=20,
        max_bins=64
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

    - static: fit one GBM on first `init_train_n` months, then predict all later months
    - walkforward: refit GBM each month on an expanding window (slower but more adaptive)
    """
    n = len(y)
    preds, truths, pred_months = [], [], []

    if mode == "static":
        # Fit once on initial window
        model = make_gbm()
        model.fit(X[:init_train_n], y[:init_train_n])

        for t in range(init_train_n, n):
            y_hat = float(model.predict(X[t:t+1])[0])
            preds.append(y_hat)
            truths.append(float(y[t]))
            pred_months.append(str(months[t]))

    elif mode == "walkforward":
        # Refit each month with fixed hyperparams (no grid search)
        model = None
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


def train_one_file_gb(
    csv_path: str,
    out_models: str,
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

    if decile not in df.columns:
        raise ValueError(f"{decile} column not found in {csv_path}")

    # --- Define RETURN target: next-month return ---
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
        X=X,
        y=y,
        months=months,
        init_train_n=init_train_n,
        mode=mode,
    )

    start_ym = df["date"].iloc[0].to_period("M")
    end_ym = df["date"].iloc[-1].to_period("M")
    train_end_ym = df["date"].iloc[init_train_n - 1].to_period("M")

    print(f"\n=== {decile} (GB RETURN, mode={mode}) ===")
    print(f"File: {csv_path}")
    print(f"Date range: {start_ym} → {end_ym}")
    print(f"Initial TRAIN: {start_ym} → {train_end_ym}  ({init_train_n} months)")
    print(f"OOS months: {len(wf['months'])} (through {wf['months'][-1]})")
    r2_str = "n/a" if wf["r2"] is None else f"{wf['r2']:.4f}"
    print(f"OOS R² (return): {r2_str} | MAE: {wf['mae']:.6f} | RMSE: {wf['rmse']:.6f}")

    # --- Save predictions ---
    os.makedirs(out_preds, exist_ok=True)
    preds_path = os.path.join(out_preds, f"{decile}_gb_oos_preds.csv")
    pd.DataFrame({
        "month": wf["months"],
        "y_true": wf["y_true"],   # true next-month return
        "y_pred": wf["y_pred"],   # predicted next-month return
    }).to_csv(preds_path, index=False)

    # --- Save model (last fitted model in chosen mode) ---
    os.makedirs(out_models, exist_ok=True)
    model_path = os.path.join(out_models, f"{decile}_gb_{mode}.pkl")
    joblib.dump(
        {
            "model": wf["model"],
            "feature_cols": feature_cols,
            "init_train_n": init_train_n,
            "mode": mode,
        },
        model_path,
    )

    # --- Save report ---
    os.makedirs(out_reports, exist_ok=True)
    report_path = os.path.join(out_reports, f"{decile}_gb_{mode}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"decile: {decile}\n")
        f.write(f"file: {csv_path}\n")
        f.write(f"date_range: {start_ym} → {end_ym}\n")
        f.write(f"initial_train_months: {init_train_n} (through {train_end_ym})\n")
        f.write(f"mode: {mode}\n")
        f.write(f"oos_months: {len(wf['months'])} (through {wf['months'][-1]})\n")
        f.write(f"oos_r2: {wf['r2']}\n")
        f.write(f"oos_mae: {wf['mae']}\n")
        f.write(f"oos_rmse: {wf['rmse']}\n")

    return {
        "decile": decile,
        "model_path": model_path,
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
        "--models_dir",
        default="models_gb",
        help="Where to save fitted GB models.",
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
    parser.add_argument(
        "--mode",
        choices=["static", "walkforward"],
        default="static",   # FAST by default
        help="static = fit once; walkforward = refit each month (slower).",
    )

    args = parser.parse_args()

    files = sorted(glob.glob(args.glob))
    if len(files) == 0:
        raise FileNotFoundError(f"No files matched: {args.glob}")

    print(f"Found {len(files)} feature files for GB RETURN training:")
    for fpath in files:
        print(" -", fpath)

    errors = []
    for fpath in files:
        try:
            _ = train_one_file_gb(
                csv_path=fpath,
                out_models=args.models_dir,
                out_preds=args.preds_dir,
                out_reports=args.reports_dir,
                prefer_train_months=args.train_months,
                mode=args.mode,
            )
        except Exception as e:
            errors.append((fpath, str(e)))
            print(f"\n[ERROR] {fpath} -> {e}\n")

    if errors:
        os.makedirs("results", exist_ok=True)
        with open("results/train_gb_errors.txt", "w", encoding="utf-8") as f:
            for path, msg in errors:
                f.write(f"{path}: {msg}\n")
        print("Some GB files failed. See → results/train_gb_errors.txt")


if __name__ == "__main__":
    main()
