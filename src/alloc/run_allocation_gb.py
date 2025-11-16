# src/alloc/run_allocation_gb.py
# Mean-variance allocator using *GB* predicted returns + volatility.
#
# Uses:
#   - GB predicted returns : results/gb_panels_returns/gb_oos_returns_panel.csv
#   - GB predicted "vol"   : results/gb_panels_vol/gb_oos_vol_panel.csv
#   - realized returns     : results/oos_panel/true_panel.csv  (same as baseline)
#
# Writes aligned semicolon CSVs with weight caps & shrinkage covariance,
# same logic as run_allocation.py but for the GB panels.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- paths & hyperparameters ----------
REL_PREDS_RET  = Path("results/gb_panels_returns/gb_oos_returns_panel.csv")
REL_PREDS_VOL  = Path("results/gb_panels_vol/gb_oos_vol_panel.csv")
REL_TRUE_RET   = Path("results/oos_panel/true_panel.csv")  # same realized panel

REL_OUT_DIR    = Path("results/alloc_gb")
REL_OUT_CSV    = REL_OUT_DIR / "weights_gb_mv.csv"
REL_OUT_META   = REL_OUT_DIR / "weights_gb_mv_meta.json"

WEIGHT_CAP     = 0.40
COV_WINDOW     = 120
MIN_COV_OBS    = 24
SHRINKAGE      = 0.50
VOL_BLEND      = 0.50
EPS_VAR        = 1e-6


# ---------- pretty CSV writer (semicolon, dot decimals, aligned columns) ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    return f"{float(val):.{decimals}f}"


def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";"):
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


# ---------- loader for GB panels (they have a 'date' column) ----------
def load_me_panel_gb(path: Path, date_col: str) -> pd.DataFrame:
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

    if date_col not in df.columns:
        raise ValueError(f"{path} must have a '{date_col}' column")

    deciles = [c for c in df.columns if c.startswith("ME")]
    ordered = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in deciles]
    if len(ordered) == 0:
        raise ValueError(f"No ME columns found in {path}")

    for c in ordered:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize to 'month' column as YYYY-MM string
    df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").astype(str)
    return df[["month"] + ordered].copy()


# ---------- loader for realized panel (same as your baseline load_me_panel) ----------
def load_me_panel_true(path: Path) -> pd.DataFrame:
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


# ---------- projection onto capped simplex ----------
def project_capped_simplex(w0: np.ndarray, cap: float) -> np.ndarray:
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


# ---------- covariance builder (same logic as baseline) ----------
def build_covariance(hist_rets: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    n = pred_var.size

    if hist_rets.shape[0] < MIN_COV_OBS:
        diag = np.maximum(pred_var, EPS_VAR)
        return np.diag(diag)

    S = np.cov(hist_rets, rowvar=False, ddof=1)
    if S.shape != (n, n):
        S = np.atleast_2d(S)
        if S.shape != (n, n):
            diag = np.maximum(pred_var, EPS_VAR)
            return np.diag(diag)

    diagS = np.diag(np.diag(S))
    Sigma = (1.0 - SHRINKAGE) * S + SHRINKAGE * diagS

    diagSigma = np.diag(Sigma)
    blended_diag = VOL_BLEND * diagSigma + (1.0 - VOL_BLEND) * np.maximum(pred_var, EPS_VAR)
    Sigma[np.diag_indices_from(Sigma)] = blended_diag

    Sigma[np.diag_indices_from(Sigma)] += EPS_VAR
    return Sigma


# ---------- mean-variance weights ----------
def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, cap: float) -> np.ndarray:
    n = mu.size
    if np.allclose(mu, 0) or np.isnan(mu).any():
        return np.full(n, 1.0 / n)

    try:
        w_dir = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        diag = np.diag(Sigma)
        inv_vol = 1.0 / np.sqrt(np.maximum(diag, EPS_VAR))
        w_dir = mu * inv_vol

    if np.allclose(w_dir, 0) or np.isnan(w_dir).any():
        return np.full(n, 1.0 / n)

    w_dir = np.clip(w_dir, 0.0, None)
    return project_capped_simplex(w_dir, cap)


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
    root        = detect_project_root()
    preds_ret_p = root / REL_PREDS_RET
    preds_vol_p = root / REL_PREDS_VOL
    true_ret_p  = root / REL_TRUE_RET
    out_dir     = root / REL_OUT_DIR
    out_csv     = root / REL_OUT_CSV
    out_meta    = root / REL_OUT_META

    out_dir.mkdir(parents=True, exist_ok=True)

    # load panels:
    # GB panels: index column is "date" -> normalize to "month"
    preds_ret = load_me_panel_gb(preds_ret_p, date_col="date")
    preds_vol = load_me_panel_gb(preds_vol_p, date_col="date")
    # realized panel already has "month"
    true_ret  = load_me_panel_true(true_ret_p)

    deciles = [f"ME{i}" for i in range(1, 11)]

    preds_ret = preds_ret.rename(columns={d: f"{d}_ret" for d in deciles})
    preds_vol = preds_vol.rename(columns={d: f"{d}_vol" for d in deciles})
    true_ret  = true_ret.rename(columns={d: f"{d}_true" for d in deciles})

    # align all by month
    df = preds_ret.merge(preds_vol, on="month", how="inner").merge(true_ret, on="month", how="inner")
    df = df.sort_values("month").reset_index(drop=True)

    n_months = df.shape[0]
    if n_months == 0:
        raise ValueError("No overlapping months between GB preds_ret, GB preds_vol, and true_ret panels.")

    rows = []
    for idx in range(n_months):
        month = df.loc[idx, "month"]

        mu = np.array([df.loc[idx, f"{d}_ret"] for d in deciles], dtype=float)
        # Treat GB vol predictions as risk proxy (no squaring; they’re already “vol-like”)
        pred_var = np.array([df.loc[idx, f"{d}_vol"] for d in deciles], dtype=float)
        pred_var = np.maximum(pred_var, EPS_VAR)

        start_idx = max(0, idx - COV_WINDOW)
        hist_slice = df.iloc[start_idx:idx]  # up to but not including idx
        if hist_slice.shape[0] > 0:
            hist_rets = np.column_stack([hist_slice[f"{d}_true"].values for d in deciles])
        else:
            hist_rets = np.empty((0, len(deciles)))

        Sigma = build_covariance(hist_rets, pred_var)
        w = mean_variance_weights(mu, Sigma, WEIGHT_CAP)

        rows.append({"month": month, **{d: float(w[i]) for i, d in enumerate(deciles)}})

    W = pd.DataFrame(rows, columns=["month"] + deciles)

    # sanity checks
    if W[deciles].isna().any().any():
        raise AssertionError("NaNs detected in GB weights.")
    sums = W[deciles].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        W[deciles] = W[deciles].div(W[deciles].sum(axis=1), axis=0)
        sums = W[deciles].sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise AssertionError("Row GB weights do not sum to 1.0 after renormalization.")

    # Write aligned CSV + meta
    write_aligned_csv(W, out_csv, decimals=6, sep=";")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "method": "GB mean-variance with shrinkage covariance + weight cap",
            "cap": WEIGHT_CAP,
            "inputs": {
                "gb_preds_ret_panel": str(preds_ret_p.relative_to(root)),
                "gb_preds_vol_panel": str(preds_vol_p.relative_to(root)),
                "true_ret_panel":     str(true_ret_p.relative_to(root)),
            },
            "deciles": deciles,
            "cov_window_months": COV_WINDOW,
            "min_cov_obs": MIN_COV_OBS,
            "shrinkage_to_diag": SHRINKAGE,
            "vol_blend": VOL_BLEND,
            "notes": (
                "GB preds_vol_panel used as risk proxy; "
                "covariance from past realized returns with diagonal shrinkage; "
                "unconstrained direction Σ^{-1}μ projected onto capped simplex."
            ),
        }, f, indent=2)

    print(f"Saved GB weights (mean-variance) → {out_csv}")
    print(f"Saved GB meta                     → {out_meta}")
    print(f"Months in GB allocation panel     : {n_months}")


if __name__ == "__main__":
    main()
