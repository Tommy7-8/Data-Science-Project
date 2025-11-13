# src/alloc/run_allocation.py
# Final allocator — mean-variance with shrinkage covariance + weight caps.
# Uses:
#   - predicted returns   : results/oos_panel/preds_panel.csv
#   - predicted volatility: results/oos_vol_panel/preds_vol_panel.csv (squared returns)
#   - realized returns    : results/oos_panel/true_panel.csv
# Writes aligned semicolon CSVs.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- paths & hyperparameters ----------
REL_PREDS_RET  = Path("results/oos_panel/preds_panel.csv")
REL_PREDS_VOL  = Path("results/oos_vol_panel/preds_vol_panel.csv")
REL_TRUE_RET   = Path("results/oos_panel/true_panel.csv")

REL_OUT_DIR    = Path("results/alloc")
REL_OUT_CSV    = REL_OUT_DIR / "weights_baseline.csv"        # same filename as before
REL_OUT_META   = REL_OUT_DIR / "weights_baseline_meta.json"  # same filename as before

WEIGHT_CAP     = 0.40   # per-asset cap (10 * 0.40 >= 1 -> feasible)
COV_WINDOW     = 120    # lookback window in months for covariance
MIN_COV_OBS    = 24     # minimum months to estimate covariance
SHRINKAGE      = 0.50   # shrinkage intensity toward diagonal
VOL_BLEND      = 0.50   # how much to blend predicted diag into covariance diag
EPS_VAR        = 1e-6   # floor for variances


# ---------- pretty CSV writer (semicolon, dot decimals, aligned columns) ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    return f"{float(val):.{decimals}f}"

def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";"):
    # left-align month; right-align numeric columns
    df_str = df.copy()
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # format numeric-looking columns
    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(lambda v: _fmt_num(v, decimals) if pd.notna(v) else "")
        else:
            df_str[c] = df_str[c].astype(str)

    df_str = df_str.astype(str)
    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    header_cells = []
    for c in df_str.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]
    for _, row in df_str.iterrows():
        cells = []
        for c in df_str.columns:
            s = row[c]
            if c in left_cols:
                cells.append(f"{s:<{widths[c]}}")
            else:
                cells.append(f"{s:>{widths[c]}}")
        lines.append(sep.join(cells))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- generic loader for ME1..ME10 semicolon panels ----------
def load_me_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", engine="python")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    if "month" not in df.columns:
        raise ValueError(f"{path} must have a 'month' column")

    deciles = [c for c in df.columns if c.startswith("ME")]
    ordered = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in deciles]
    if len(ordered) != 10:
        raise ValueError(f"Expected 10 ME columns in {path}, found: {sorted(deciles)}")

    for c in ordered:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["month"] + ordered].copy()


# ---------- projection onto capped simplex (long-only, sum=1, w_i <= cap) ----------
def project_capped_simplex(w0: np.ndarray, cap: float) -> np.ndarray:
    """
    Project nonnegative w0 onto {w >= 0, sum w = 1, w_i <= cap} using a water-filling scheme.
    Guarantees: long-only, row sum == 1, each weight <= cap, no NaNs.
    """
    w0 = np.clip(w0, 0.0, None)
    n = w0.size
    cap = float(cap)
    if n * cap < 1.0:
        cap = 1.0 / n  # ensure feasibility

    s = w0.sum()
    w0 = np.full(n, 1.0 / n) if s <= 0 else (w0 / s)

    fixed = np.zeros(n, dtype=bool)
    w = np.zeros(n)

    while True:
        free = ~fixed
        budget = 1.0 - fixed.sum() * cap
        if budget < 0:
            budget = 0.0

        if free.sum() == 0:
            w[fixed] = cap
            tot = w.sum()
            w = (w / tot) if tot > 0 else np.full(n, 1.0 / n)
            return np.clip(w, 0.0, cap)

        denom = w0[free].sum()
        if denom <= 0:
            w[free] = budget / free.sum()
        else:
            w[free] = (w0[free] / denom) * budget
        w[fixed] = cap

        over = (w > cap) & free
        if not over.any():
            tot = w.sum()
            w = (w / tot) if tot > 0 else np.full(n, 1.0 / n)
            return np.clip(w, 0.0, cap)

        fixed[over] = True
        w[over] = 0.0  # will be set to cap next loop


# ---------- covariance builder with shrinkage + predicted diag ----------
def build_covariance(hist_rets: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    n = pred_var.size

    # if not enough history, fall back to diagonal from predicted variance
    if hist_rets.shape[0] < MIN_COV_OBS:
        diag = np.maximum(pred_var, EPS_VAR)
        return np.diag(diag)

    # sample covariance
    S = np.cov(hist_rets, rowvar=False, ddof=1)
    if S.shape != (n, n):
        S = np.atleast_2d(S)
        if S.shape != (n, n):
            diag = np.maximum(pred_var, EPS_VAR)
            return np.diag(diag)

    # shrink toward diagonal
    diagS = np.diag(np.diag(S))
    Sigma = (1.0 - SHRINKAGE) * S + SHRINKAGE * diagS

    # blend predicted variance into diagonal
    diagSigma = np.diag(Sigma)
    blended_diag = VOL_BLEND * diagSigma + (1.0 - VOL_BLEND) * np.maximum(pred_var, EPS_VAR)
    Sigma[np.diag_indices_from(Sigma)] = blended_diag

    # small diagonal bump for numerical stability
    Sigma[np.diag_indices_from(Sigma)] += EPS_VAR

    return Sigma


# ---------- simple mean-variance weight from mu and Sigma ----------
def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, cap: float) -> np.ndarray:
    n = mu.size
    if np.allclose(mu, 0) or np.isnan(mu).any():
        return np.full(n, 1.0 / n)

    # unconstrained direction: Sigma^{-1} mu
    try:
        w_dir = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        # fall back: risk-adjusted mu using diagonal only
        diag = np.diag(Sigma)
        inv_vol = 1.0 / np.sqrt(np.maximum(diag, EPS_VAR))
        w_dir = mu * inv_vol

    if np.allclose(w_dir, 0) or np.isnan(w_dir).any():
        return np.full(n, 1.0 / n)

    # enforce long-only + caps via projection
    w_dir = np.clip(w_dir, 0.0, None)
    w = project_capped_simplex(w_dir, cap)
    return w


# ---------- project root detection ----------
def detect_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / REL_PREDS_RET).exists():
        return cwd
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / REL_PREDS_RET).exists():
            return p
    return cwd


# ---------- main ----------
def main():
    root       = detect_project_root()
    preds_ret_p = root / REL_PREDS_RET
    preds_vol_p = root / REL_PREDS_VOL
    true_ret_p  = root / REL_TRUE_RET
    out_dir    = root / REL_OUT_DIR
    out_csv    = root / REL_OUT_CSV
    out_meta   = root / REL_OUT_META

    out_dir.mkdir(parents=True, exist_ok=True)

    # load panels
    preds_ret = load_me_panel(preds_ret_p)   # predicted returns
    preds_vol = load_me_panel(preds_vol_p)   # predicted "vol" (squared returns)
    true_ret  = load_me_panel(true_ret_p)    # realized returns

    # rename columns to keep them separate
    deciles = [f"ME{i}" for i in range(1, 11)]
    preds_ret = preds_ret.rename(columns={d: f"{d}_ret" for d in deciles})
    preds_vol = preds_vol.rename(columns={d: f"{d}_vol" for d in deciles})
    true_ret  = true_ret.rename(columns={d: f"{d}_true" for d in deciles})

    # align all by month
    df = preds_ret.merge(preds_vol, on="month", how="inner").merge(true_ret, on="month", how="inner")
    df = df.sort_values("month").reset_index(drop=True)

    n_months = df.shape[0]
    if n_months == 0:
        raise ValueError("No overlapping months between preds_ret, preds_vol, and true_ret panels.")

    rows = []
    for idx in range(n_months):
        month = df.loc[idx, "month"]

        # predicted mean returns and predicted variance (from vol model)
        mu = np.array([df.loc[idx, f"{d}_ret"] for d in deciles], dtype=float)
        pred_var = np.array([df.loc[idx, f"{d}_vol"] for d in deciles], dtype=float)
        pred_var = np.maximum(pred_var, EPS_VAR)

        # build historical window for covariance using realized returns up to t-1
        start_idx = max(0, idx - COV_WINDOW)
        hist_slice = df.iloc[start_idx:idx]    # up to but not including idx
        if hist_slice.shape[0] > 0:
            hist_rets = np.column_stack([hist_slice[f"{d}_true"].values for d in deciles])
        else:
            hist_rets = np.empty((0, len(deciles)))

        Sigma = build_covariance(hist_rets, pred_var)
        w = mean_variance_weights(mu, Sigma, WEIGHT_CAP)

        rows.append({"month": month, **{d: float(w[i]) for i, d in enumerate(deciles)}})

    W = pd.DataFrame(rows, columns=["month"] + deciles)

    # sanity: row sums ~ 1 and no NaNs
    if W[deciles].isna().any().any():
        raise AssertionError("NaNs detected in weights.")
    sums = W[deciles].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        # final renormalization if needed
        W[deciles] = W[deciles].div(W[deciles].sum(axis=1), axis=0)
        sums = W[deciles].sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise AssertionError("Row weights do not sum to 1.0 after renormalization.")

    # Write aligned CSV + meta
    write_aligned_csv(W, out_csv, decimals=6, sep=";")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "method": "mean-variance with shrinkage covariance + weight cap",
            "cap": WEIGHT_CAP,
            "inputs": {
                "preds_ret_panel": str(preds_ret_p.relative_to(root)),
                "preds_vol_panel": str(preds_vol_p.relative_to(root)),
                "true_ret_panel":  str(true_ret_p.relative_to(root))
            },
            "deciles": deciles,
            "cov_window_months": COV_WINDOW,
            "min_cov_obs": MIN_COV_OBS,
            "shrinkage_to_diag": SHRINKAGE,
            "vol_blend": VOL_BLEND,
            "notes": (
                "preds_vol_panel used as forecast of squared returns (risk proxy); "
                "covariance from past realized returns with diagonal shrinkage; "
                "unconstrained direction Σ^{-1}μ projected onto capped simplex."
            )
        }, f, indent=2)

    print(f"Saved weights (mean-variance) → {out_csv}")
    print(f"Saved meta                     → {out_meta}")
    print(f"Months in allocation panel     : {n_months}")


if __name__ == "__main__":
    main()
