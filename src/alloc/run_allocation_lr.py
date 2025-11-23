# src/alloc/run_allocation_lr.py
# Final LR allocator — mean-variance with shrinkage covariance + weight caps.
#
# Uses LR (Ridge) models as inputs:
#   - predicted returns   : results/oos_panel_lr/preds_panel_lr.csv
#   - predicted volatility: results/oos_vol_panel_lr/preds_vol_panel_lr.csv (predicted squared returns)
#   - realized returns    : results/oos_panel_lr/true_panel_lr.csv
#
# Output:
#   - results/alloc_lr/weights_baseline.csv        : monthly ME1..ME10 portfolio weights
#   - results/alloc_lr/weights_baseline_meta.json  : metadata & hyperparameters
#
# All CSVs are semicolon-separated and visually aligned.

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- paths & hyperparameters ----------

REL_PREDS_RET = Path("results/oos_panel_lr/preds_panel_lr.csv")
REL_PREDS_VOL = Path("results/oos_vol_panel_lr/preds_vol_panel_lr.csv")
REL_TRUE_RET = Path("results/oos_panel_lr/true_panel_lr.csv")

REL_OUT_DIR = Path("results/alloc_lr")
REL_OUT_CSV = REL_OUT_DIR / "weights_baseline.csv"        # LR weights
REL_OUT_META = REL_OUT_DIR / "weights_baseline_meta.json"  # LR meta

# Allocation knobs
WEIGHT_CAP = 0.40   # per-asset cap (10 * 0.40 >= 1 -> feasible)
COV_WINDOW = 120    # lookback window in months for covariance
MIN_COV_OBS = 24    # minimum months to estimate covariance from history
SHRINKAGE = 0.50    # shrinkage intensity toward diagonal
VOL_BLEND = 0.50    # blend between sample diag and predicted diag
EPS_VAR = 1e-6      # floor for variances (numerical stability)


# ---------- pretty CSV writer (semicolon, aligned columns, trimmed zeros) ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """
    Format numbers with up to `decimals` decimal places,
    removing unnecessary trailing zeros and trailing decimal points.

    Returns an empty string for NaN.
    """
    if pd.isna(val):
        return ""
    s = f"{float(val):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def write_aligned_csv(
    df: pd.DataFrame,
    path: Path,
    decimals: int = 6,
    sep: str = ";",
) -> None:
    """
    Write a DataFrame as a semicolon-separated, visually aligned CSV.

    Layout:
      - 'month' column left-aligned
      - all other columns right-aligned
      - numeric-looking cells formatted via _fmt_num

    This matches the formatting used in the other scripts (panels, predictions, etc.).
    """
    df_str = df.copy()

    # 'month' is the only left-aligned column
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-ish columns on the right
    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(
                lambda v: _fmt_num(v, decimals) if pd.notna(v) else ""
            )
        else:
            df_str[c] = df_str[c].astype(str)

    # Compute widths
    df_str = df_str.astype(str)
    widths = {
        c: max(len(c), df_str[c].map(len).max())
        for c in df_str.columns
    }

    # Header row
    header_cells = []
    for c in df_str.columns:
        if c in left_cols:
            header_cells.append(f"{c:<{widths[c]}}")
        else:
            header_cells.append(f"{c:>{widths[c]}}")

    lines = [sep.join(header_cells)]

    # Data rows
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
    """
    Load a panel of ME1..ME10 series from an aligned semicolon CSV (or plain CSV).

    Expected structure:
      - 'month' column
      - ME1..ME10 columns (10 deciles, percent/decimal returns or vol)

    Steps:
      - handle both semicolon-separated and comma-separated files
      - strip spaces and BOM from headers and cells
      - convert ME columns to numeric

    Returns:
        DataFrame with columns ['month', 'ME1', ..., 'ME10'].
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    # Detect aligned ; format vs normal ,
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            encoding="utf-8-sig",
            engine="python",
        )
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    if "month" not in df.columns:
        raise ValueError(f"{path} must have a 'month' column")

    # Identify ME decile columns and enforce ME1..ME10 ordering
    deciles = [c for c in df.columns if c.startswith("ME")]
    ordered = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in deciles]
    if len(ordered) != 10:
        raise ValueError(f"Expected 10 ME columns in {path}, found: {sorted(deciles)}")

    # Convert ME columns to numeric
    for c in ordered:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[["month"] + ordered].copy()


# ---------- projection onto capped simplex (long-only, sum=1, w_i <= cap) ----------

def project_capped_simplex(w0: np.ndarray, cap: float) -> np.ndarray:
    """
    Project a weight vector onto the capped simplex:

        { w >= 0, sum(w) = 1, w_i <= cap for all i }

    The algorithm is a simple water-filling scheme:

      1. Clip negatives and renormalize w0 to sum to 1 (or use uniform).
      2. Iteratively:
         - allocate budget across free (non-capped) assets
         - if any exceed cap, fix them at cap and repeat on the rest.

    Guarantees:
      - long-only weights (>= 0)
      - each weight <= cap
      - sum of weights == 1 up to numerical tolerance
    """
    w0 = np.clip(w0, 0.0, None)
    n = w0.size
    cap = float(cap)

    # If cap is too small to reach sum 1, relax it to 1/n (just in case).
    if n * cap < 1.0:
        cap = 1.0 / n

    s = w0.sum()
    if s <= 0:
        w0 = np.full(n, 1.0 / n)
    else:
        w0 = w0 / s

    fixed = np.zeros(n, dtype=bool)
    w = np.zeros(n)

    while True:
        free = ~fixed
        # Budget for free weights: 1 minus sum of all fixed caps
        budget = 1.0 - fixed.sum() * cap
        if budget < 0:
            budget = 0.0

        if free.sum() == 0:
            # Everything is fixed at cap; renormalize just in case
            w[fixed] = cap
            tot = w.sum()
            w = (w / tot) if tot > 0 else np.full(n, 1.0 / n)
            return np.clip(w, 0.0, cap)

        denom = w0[free].sum()
        if denom <= 0:
            w[free] = budget / free.sum()
        else:
            w[free] = (w0[free] / denom) * budget

        # Fixed ones are pinned at cap
        w[fixed] = cap

        # Check if any free weights violate cap
        over = (w > cap) & free
        if not over.any():
            # All constraints satisfied → final normalization
            tot = w.sum()
            w = (w / tot) if tot > 0 else np.full(n, 1.0 / n)
            return np.clip(w, 0.0, cap)

        # Fix those exceeding cap and continue
        fixed[over] = True
        w[over] = 0.0  # will be set to cap in the next loop


# ---------- covariance builder with shrinkage + predicted diag ----------

def build_covariance(hist_rets: np.ndarray, pred_var: np.ndarray) -> np.ndarray:
    """
    Build a covariance matrix for the ME deciles.

    Ingredients:
      - hist_rets: matrix of realized returns from the past window (T × N).
      - pred_var: vector of predicted variances (squared returns) from the vol model.

    Logic:
      1. If not enough history (rows < MIN_COV_OBS), use purely diagonal covariance
         based on predicted variances.
      2. Otherwise:
           - compute sample covariance S
           - shrink towards its diagonal: Sigma = (1-λ)S + λ diag(S)
           - blend diagonal with predicted variances using VOL_BLEND
           - add a small diagonal bump EPS_VAR for stability
    """
    n = pred_var.size

    # Not enough observations: fallback to diagonal using predicted variance only
    if hist_rets.shape[0] < MIN_COV_OBS:
        diag = np.maximum(pred_var, EPS_VAR)
        return np.diag(diag)

    # Sample covariance (columns are assets)
    S = np.cov(hist_rets, rowvar=False, ddof=1)
    if S.shape != (n, n):
        S = np.atleast_2d(S)
        if S.shape != (n, n):
            diag = np.maximum(pred_var, EPS_VAR)
            return np.diag(diag)

    # Shrink towards its diagonal
    diagS = np.diag(np.diag(S))
    Sigma = (1.0 - SHRINKAGE) * S + SHRINKAGE * diagS

    # Blend predicted variance into the diagonal
    diagSigma = np.diag(Sigma)
    blended_diag = VOL_BLEND * diagSigma + (1.0 - VOL_BLEND) * np.maximum(pred_var, EPS_VAR)
    Sigma[np.diag_indices_from(Sigma)] = blended_diag

    # Small diagonal bump for numerical stability
    Sigma[np.diag_indices_from(Sigma)] += EPS_VAR

    return Sigma


# ---------- simple mean-variance weight from mu and Sigma ----------

def mean_variance_weights(mu: np.ndarray, Sigma: np.ndarray, cap: float) -> np.ndarray:
    """
    Compute long-only, capped mean-variance weights given expected returns and covariance.

    Steps:
      1. Compute the unconstrained direction: w_dir = Σ^{-1} μ.
         If Σ is singular, fall back to a diagonal risk adjustment.
      2. Clip to nonnegative and project onto the capped simplex
         (sum=1, each weight <= cap).

    If anything degenerates (μ all zeros / NaNs, or direction collapses),
    return the equal-weight portfolio.
    """
    n = mu.size

    if np.allclose(mu, 0) or np.isnan(mu).any():
        return np.full(n, 1.0 / n)

    # Try to solve Σw = μ
    try:
        w_dir = np.linalg.solve(Sigma, mu)
    except np.linalg.LinAlgError:
        # Fallback: diagonal-only risk scaling
        diag = np.diag(Sigma)
        inv_vol = 1.0 / np.sqrt(np.maximum(diag, EPS_VAR))
        w_dir = mu * inv_vol

    if np.allclose(w_dir, 0) or np.isnan(w_dir).any():
        return np.full(n, 1.0 / n)

    # Enforce long-only + caps
    w_dir = np.clip(w_dir, 0.0, None)
    w = project_capped_simplex(w_dir, cap)
    return w


# ---------- project root detection ----------

def detect_project_root() -> Path:
    """
    Try to locate the project root so the script works when called
    from different working directories (e.g., project root or src/).

    Strategy:
      - If results/oos_panel_lr/preds_panel_lr.csv exists under cwd, use cwd.
      - Otherwise walk up from this file's directory and pick the first parent
        where that path exists.
      - Fall back to cwd if nothing is found.
    """
    cwd = Path.cwd()
    if (cwd / REL_PREDS_RET).exists():
        return cwd

    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / REL_PREDS_RET).exists():
            return p

    return cwd


# ---------- main pipeline ----------

def main() -> None:
    """
    Run the LR-based allocation pipeline:

      1. Locate project root and all input panels.
      2. Load predicted returns, predicted vol (squared returns), and realized returns.
      3. For each month:
           - build covariance from a rolling window of past realized returns
           - use mean-variance with shrinkage and caps to get weights
      4. Save the panel of weights and a JSON file with metadata.
    """
    root = detect_project_root()
    preds_ret_p = root / REL_PREDS_RET
    preds_vol_p = root / REL_PREDS_VOL
    true_ret_p = root / REL_TRUE_RET

    out_dir = root / REL_OUT_DIR
    out_csv = root / REL_OUT_CSV
    out_meta = root / REL_OUT_META

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load LR panels:
    #   - predicted returns (μ)
    #   - predicted "volatility" (squared returns)
    #   - realized returns (for covariance)
    preds_ret = load_me_panel(preds_ret_p)   # predicted returns (LR)
    preds_vol = load_me_panel(preds_vol_p)   # predicted squared returns (LR vol model)
    true_ret = load_me_panel(true_ret_p)     # realized returns

    # Rename columns to keep types separate when merging
    deciles = [f"ME{i}" for i in range(1, 11)]
    preds_ret = preds_ret.rename(columns={d: f"{d}_ret" for d in deciles})
    preds_vol = preds_vol.rename(columns={d: f"{d}_vol" for d in deciles})
    true_ret = true_ret.rename(columns={d: f"{d}_true" for d in deciles})

    # Align all on 'month' and sort
    df = (
        preds_ret
        .merge(preds_vol, on="month", how="inner")
        .merge(true_ret, on="month", how="inner")
    )
    df = df.sort_values("month").reset_index(drop=True)

    n_months = df.shape[0]
    if n_months == 0:
        raise ValueError(
            "No overlapping months between preds_ret, preds_vol, and true_ret panels."
        )

    rows = []
    for idx in range(n_months):
        month = df.loc[idx, "month"]

        # Predicted mean returns and predicted variances for each decile
        mu = np.array(
            [df.loc[idx, f"{d}_ret"] for d in deciles],
            dtype=float,
        )
        pred_var = np.array(
            [df.loc[idx, f"{d}_vol"] for d in deciles],
            dtype=float,
        )
        pred_var = np.maximum(pred_var, EPS_VAR)

        # Build historical window of realized returns up to t-1 (for covariance)
        start_idx = max(0, idx - COV_WINDOW)
        hist_slice = df.iloc[start_idx:idx]  # up to but not including idx
        if hist_slice.shape[0] > 0:
            hist_rets = np.column_stack(
                [hist_slice[f"{d}_true"].values for d in deciles]
            )
        else:
            hist_rets = np.empty((0, len(deciles)))

        Sigma = build_covariance(hist_rets, pred_var)
        w = mean_variance_weights(mu, Sigma, WEIGHT_CAP)

        rows.append({
            "month": month,
            **{d: float(w[i]) for i, d in enumerate(deciles)},
        })

    # Assemble weights panel
    W = pd.DataFrame(rows, columns=["month"] + deciles)

    # Sanity checks: no NaNs and row sums ≈ 1
    if W[deciles].isna().any().any():
        raise AssertionError("NaNs detected in weights.")

    sums = W[deciles].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-9):
        # Final renormalization if needed
        W[deciles] = W[deciles].div(W[deciles].sum(axis=1), axis=0)
        sums = W[deciles].sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise AssertionError("Row weights do not sum to 1.0 after renormalization.")

    # Write aligned CSV of weights
    write_aligned_csv(W, out_csv, decimals=6, sep=";")

    # Write metadata JSON with all key parameters
    meta = {
        "method": "LR mean-variance with shrinkage covariance + weight cap",
        "cap": WEIGHT_CAP,
        "inputs": {
            "preds_ret_panel": str(preds_ret_p.relative_to(root)),
            "preds_vol_panel": str(preds_vol_p.relative_to(root)),
            "true_ret_panel": str(true_ret_p.relative_to(root)),
        },
        "deciles": deciles,
        "cov_window_months": COV_WINDOW,
        "min_cov_obs": MIN_COV_OBS,
        "shrinkage_to_diag": SHRINKAGE,
        "vol_blend": VOL_BLEND,
        "notes": (
            "preds_vol_panel used as forecast of squared returns (risk proxy); "
            "covariance from past realized returns with diagonal shrinkage; "
            "unconstrained direction Sigma^{-1} * mu projected onto capped simplex."
        ),
    }

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Console summary
    print(f"Saved LR weights (mean-variance) → {out_csv}")
    print(f"Saved meta                      → {out_meta}")
    print(f"Months in allocation panel      : {n_months}")


if __name__ == "__main__":
    main()
