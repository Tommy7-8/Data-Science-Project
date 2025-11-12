# src/models/explain_one_prediction.py
# Reproduce and EXPLAIN a single prediction (ME1, 1947-08 by default).
# It refits ONLY up to the month before the target (expanding window),
# using the SAME logic as train_all.py, then prints every number:
# x (raw), mean, std, z = (x-mean)/std, coef, contrib = coef*z, intercept, and the final sum.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# ------- CONFIG (edit if you want to explain a different decile/month) -------
DECILE = "ME1"
TARGET_MONTH = "1947-08"  # month being predicted
FEATURES_CSV = Path("data/processed") / f"features_{DECILE}_full.csv"
ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
PREFER_TRAIN_MONTHS = 240  # ~20y
MIN_TRAIN = 120
# ---------------------------------------------------------------------------

def select_lagged_features(df: pd.DataFrame, decile_prefix: str) -> List[str]:
    cols = df.columns.tolist()
    lag_cols = [c for c in cols if c.startswith(f"{decile_prefix}_lag")]
    vol_cols = [c for c in cols if c.startswith(f"{decile_prefix}_vol_")]
    factor_lag_cols = [c for c in cols if c.endswith("_lag1")]
    seen, ordered = set(), []
    for c in lag_cols + vol_cols + factor_lag_cols:
        if c in df.columns and c not in seen:
            seen.add(c); ordered.append(c)
    return ordered

def first_train_size(n_obs: int, prefer: int = PREFER_TRAIN_MONTHS, min_train: int = MIN_TRAIN) -> int:
    return max(min_train, min(prefer, n_obs - 1))

def fit_ridge_cv(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[Pipeline, dict]:
    # same scaler+ridge and CV schedule as train_all.py
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(random_state=0))
    ])
    # keep CV small and time-respecting
    n_splits = max(2, min(5, max(2, X_train.shape[0] // 60)))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gscv = GridSearchCV(
        pipe,
        param_grid={"ridge__alpha": ALPHA_GRID},
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        refit=True,
        verbose=0
    )
    gscv.fit(X_train, y_train)
    return gscv.best_estimator_, gscv.best_params_

def main():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing features file: {FEATURES_CSV}")

    # Load features for the chosen decile
    df = pd.read_csv(FEATURES_CSV)
    if "date" not in df.columns or DECILE not in df.columns:
        raise ValueError("Expected 'date' column and the decile column in the features CSV.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Build the target (next-month return of this decile) and the target month label
    df["target"] = df[DECILE].shift(-1)
    df["target_month"] = df["date"].shift(-1).dt.to_period("M").astype(str)
    df = df.dropna(subset=["target", "target_month"]).copy()

    # Feature set
    feature_cols = select_lagged_features(df, decile_prefix=DECILE)
    if not feature_cols:
        raise RuntimeError(f"No lagged features found for {DECILE}.")

    X = df[feature_cols].astype(float).to_numpy()
    y = df["target"].astype(float).to_numpy()
    months = df["target_month"].to_numpy()

    # Initial train size (~20y), then identify the row t that predicts TARGET_MONTH
    n = len(df)
    init_train_n = first_train_size(n)
    # Find the index t where months[t] == TARGET_MONTH
    try:
        t = int(np.where(months == TARGET_MONTH)[0][0])
    except IndexError:
        raise ValueError(f"TARGET_MONTH={TARGET_MONTH} not found in this file.")

    if t < init_train_n:
        raise ValueError(f"TARGET_MONTH='{TARGET_MONTH}' occurs before the first OOS index ({init_train_n}).")

    # Fit ONLY on history up to t-1 (expanding window), exactly like the walk-forward step
    X_train, y_train = X[:t], y[:t]
    model, best = fit_ridge_cv(X_train, y_train)

    # Extract components
    scaler: StandardScaler = model.named_steps["scaler"]
    ridge: Ridge = model.named_steps["ridge"]
    mu = scaler.mean_              # training means per feature
    sigma = scaler.scale_          # training stds per feature
    beta = ridge.coef_             # coefficients on standardized features
    intercept = ridge.intercept_   # intercept (already accounting for standardization)
    x_raw = X[t, :]                # the July-1947 raw feature vector

    # Standardize this row using *training* mean/std
    z = (x_raw - mu) / sigma
    contrib = beta * z
    y_hat = float(intercept + contrib.sum())

    # Print everything nicely
    print(f"\nExplain prediction for {DECILE}, TARGET_MONTH={TARGET_MONTH}")
    print(f"Training window rows: 0 .. {t-1}  (size={len(X_train)})")
    print(f"Selected alpha: {best.get('ridge__alpha')}")
    print("\nColumns: feature | x_raw | mean | std | z | beta | beta*z (contrib)")
    rows = []
    for i, name in enumerate(feature_cols):
        rows.append([name, x_raw[i], mu[i], sigma[i], z[i], beta[i], contrib[i]])
    expl = pd.DataFrame(rows, columns=["feature","x_raw","mean","std","z","beta","beta*z"])
    # Sort by absolute contribution to see top drivers
    expl = expl.reindex(expl["beta*z"].abs().sort_values(ascending=False).index)
    with pd.option_context('display.max_rows', None, 'display.float_format', '{: .10f}'.format):
        print(expl.to_string(index=False))

    print("\nIntercept (beta0): {:.12f}".format(intercept))
    print("Sum of contributions: {:.12f}".format(contrib.sum()))
    print("Predicted return y_hat: {:.12f}".format(y_hat))

    # Optional: if the OOS file exists, compare
    preds_csv = Path("results/oos_preds") / f"{DECILE}_oos_preds.csv"
    if preds_csv.exists():
        pred_row = pd.read_csv(preds_csv)
        pred_row = pred_row.loc[pred_row["month"] == TARGET_MONTH]
        if len(pred_row):
            print("\nCross-check with saved OOS prediction:")
            print(pred_row.to_string(index=False))

if __name__ == "__main__":
    main()
