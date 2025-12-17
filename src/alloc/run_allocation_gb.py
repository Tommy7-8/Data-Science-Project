# src/alloc/run_allocation_gb.py
# Mean–variance allocator using Gradient Boosting (GB) predicted returns and
# GB predicted *volatility* (a standard-deviation-like realized-vol forecast).
#
# Inputs:
#   results/oos_panel_gb/preds_panel_gb.csv         — GB predicted returns (μ̂)
#   results/oos_vol_panel_gb/preds_vol_panel_gb.csv — GB predicted volatility (σ̂)
#   results/oos_panel_lr/true_panel_lr.csv          — realized ME1..ME10 returns (used to estimate Σ)
#
# Outputs:
#   results/alloc_gb/weights_gb_mv.csv        — monthly ME1..ME10 portfolio weights
#   results/alloc_gb/weights_gb_mv_meta.json  — metadata + settings
#
# Notes:
#   - The GB “vol” panel contains volatility, not variance. We convert via:
#       σ̂² = (max(σ̂, 0))²
#   - No transaction costs here; evaluation is handled in evaluate_allocation_gb.py.
#
# The allocator:
#   - builds Σ_t from a rolling window of realized returns, with shrinkage,
#     and a blended diagonal using σ̂²
#   - uses Σ^{-1} μ direction and projects to a long-only capped simplex

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- input/output paths ----------

REL_PREDS_RET = Path("results/oos_panel_gb/preds_panel_gb.csv")      # μ̂ from GB
REL_PREDS_VOL = Path("results/oos_vol_panel_gb/preds_vol_panel_gb.csv")  # σ̂ (GB)
REL_TRUE_RET = Path("results/oos_panel_lr/true_panel_lr.csv")        # realized returns

REL_OUT_DIR = Path("results/alloc_gb")
REL_OUT_CSV = REL_OUT_DIR / "weights_gb_mv.csv"
REL_OUT_META = REL_OUT_DIR / "weights_gb_mv_meta.json"

# ---------- allocator hyperparameters ----------

WEIGHT_CAP = 0.40     # cap on each ME decile weight
COV_WINDOW = 120       # months of past realized returns for covariance
MIN_COV_OBS = 24       # minimum months required for covariance estimation
SHRINKAGE = 0.50       # shrinkage intensity toward diagonal
VOL_BLEND = 0.50       # blend between sample diag and predicted diag
EPS_VAR = 1e-6         # floor for predicted variances


# ---------- formatting: aligned semicolon CSV ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """Format numbers with trimmed trailing zeros (consistent with all project CSVs)."""
    if pd.isna(val):
        return ""
    s = f"{float(val):.{decimals}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";") -> None:
    """
    Write DataFrame as a semicolon-separated, visually aligned CSV:
      - 'month' left-aligned
      - numeric columns right-aligned
      - trimming trailing zeros
    """
    df_str = df.copy()

    left_cols = ["month"]
    right_cols = [c for c in df_str.columns if c != "month"]

    # numeric formatting
    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        df_str[c] = ser.map(lambda v: _fmt_num(v, decimals))

    df_str = df_str.astype(str)

    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    header = [
        f"{c:<{widths[c]}}" if c == "month" else f"{c:>{widths[c]}}"
        for c in df_str.columns
    ]

    lines = [sep.join(header)]
    for _, row in df_str.iterrows():
        line = [
            f"{row[c]:<{widths[c]}}" if c == "month" else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        lines.append(sep.join(line))

    path.write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- robust panel loader ----------

def load_panel(path: Path) -> pd.DataFrame:
    """
    Load ME1..ME10 panels written in semicolon-aligned format (or fallback comma CSV).
    Converts ME columns to numeric and preserves the 'month' column.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", dtype=str, engine="python")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        # strip padding inside cells
        obj = df.select_dtypes(include="object").columns
        df[obj] = df[obj].apply(lambda s: s.str.strip())
    else:
        df = pd.read_csv(path, dtype=str)

    if "month" not in df.columns:
        raise ValueError(f"{path} missing 'month'")

    deciles = [f"ME{i}" for i in range(1, 11)]
    for d in deciles:
        df[d] = pd.to_numeric(df[d], errors="coerce")

    return df[["month"] + deciles]


# ---------- projection onto capped simplex ----------

def project_capped_simplex(w0: np.ndarray, cap: float) -> np.ndarray:
    """
    Project raw direction w0 onto:
        w_i >= 0, sum(w_i) = 1, and w_i <= cap.
    Classic water-filling algorithm.
    """
    w0 = np.clip(w0, 0, None)
    n = len(w0)

    # ensure feasibility
    if n * cap < 1:
        cap = 1 / n

    # normalize initial direction
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(n, 1 / n)

    fixed = np.zeros(n, dtype=bool)
    w = np.zeros(n)

    while True:
        free = ~fixed
        budget = 1 - fixed.sum() * cap

        if free.sum() == 0:
            w[fixed] = cap
            return w / w.sum()

        denom = w0[free].sum()
        w[free] = (w0[free] / denom) * budget if denom > 0 else budget / free.sum()
        w[fixed] = cap

        over = (w > cap) & free
        if not over.any():
            return w / w.sum()

        fixed[over] = True
        w[over] = 0.0   # will be reset to cap on next iteration


# ---------- covariance builder ----------

def build_covariance(hist: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    """
    Build covariance Σ_t:
      - if too little history: diagonal(pred_var)
      - else:
          S = sample covariance
          Σ = (1-λ) S + λ diag(S)
          diag(Σ) = blend(sample_diag, pred_var)
    """
    n = len(pred_var)

    # not enough past data
    if len(hist) < MIN_COV_OBS:
        return np.diag(np.maximum(pred_var, EPS_VAR))

    S = np.cov(hist, rowvar=False, ddof=1)
    if S.shape != (n, n):
        return np.diag(np.maximum(pred_var, EPS_VAR))

    # shrinkage
    diagS = np.diag(np.diag(S))
    Sigma = (1 - SHRINKAGE) * S + SHRINKAGE * diagS

    # blend diagonal with predicted variance (σ̂² from the vol forecast)
    diagSigma = np.diag(Sigma)
    blended = VOL_BLEND * diagSigma + (1 - VOL_BLEND) * pred_var
    np.fill_diagonal(Sigma, blended)

    return Sigma


# ---------- main ----------

def main() -> None:
    root = Path.cwd()

    # Load GB panels (predicted returns + predicted vol)
    preds_ret = load_panel(root / REL_PREDS_RET)
    preds_vol = load_panel(root / REL_PREDS_VOL)

    # Load realized returns for covariance estimation
    true_ret = load_panel(root / REL_TRUE_RET)

    dec = [f"ME{i}" for i in range(1, 11)]

    # Add suffixes to avoid collisions before merging
    preds_ret = preds_ret.rename(columns={d: f"{d}_ret" for d in dec})
    preds_vol = preds_vol.rename(columns={d: f"{d}_vol" for d in dec})
    true_ret = true_ret.rename(columns={d: f"{d}_true" for d in dec})

    # Merge into single month-aligned table
    df = (
        preds_ret
        .merge(preds_vol, on="month")
        .merge(true_ret, on="month")
        .sort_values("month")
        .reset_index(drop=True)
    )

    out_rows = []

    # Walk month by month
    for idx, row in df.iterrows():
        mu = np.array([row[f"{d}_ret"] for d in dec], float)
        pred_vol = np.array([row[f"{d}_vol"] for d in dec], float)
        # GB vol model outputs volatility (std-like). Convert to variance for Σ.
        pred_vol = np.nan_to_num(pred_vol, nan=0.0, posinf=0.0, neginf=0.0)
        pred_var = np.maximum(pred_vol, 0.0) ** 2

        # Historical window for covariance
        start = max(0, idx - COV_WINDOW)
        hist = df.iloc[start:idx][[f"{d}_true" for d in dec]].to_numpy()

        Sigma = build_covariance(hist, pred_var)

        # Unconstrained mean-variance direction
        try:
            w_dir = np.linalg.solve(Sigma, mu)
        except np.linalg.LinAlgError:
            w_dir = mu  # fallback direction

        # Projection to long-only capped weights
        w = project_capped_simplex(w_dir, WEIGHT_CAP)

        out_rows.append({
            "month": row["month"],
            **{d: float(w[i]) for i, d in enumerate(dec)}
        })

    W = pd.DataFrame(out_rows)

    # Save weights
    REL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(W, root / REL_OUT_CSV)

    # Save metadata
    meta = {
        "method": "GB mean-variance allocation",
        "weight_cap": WEIGHT_CAP,
        "inputs": {
            "gb_pred_returns": str(REL_PREDS_RET),
            "gb_pred_vol": str(REL_PREDS_VOL),
            "true_returns": str(REL_TRUE_RET)
        }
    }
    (root / REL_OUT_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved GB weights → {REL_OUT_CSV}")
    print(f"Saved GB meta    → {REL_OUT_META}")


if __name__ == "__main__":
    main()
