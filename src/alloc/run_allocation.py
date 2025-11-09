# src/alloc/run_allocation.py
# Baseline allocator — click-to-run from project root (PS ...\Data Science Project>)
# Robust capped-simplex projection; auto-detects root if ▶ is pressed from another folder.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

REL_PREDS = Path("results/oos_panel/preds_panel.csv")
REL_OUT_DIR = Path("results/alloc")
REL_OUT_CSV = REL_OUT_DIR / "weights_baseline.csv"
REL_OUT_META = REL_OUT_DIR / "weights_baseline_meta.json"
WEIGHT_CAP = 0.30  # per-asset cap (feasible for 10 names because 10*0.30 >= 1)

def detect_project_root() -> Path:
    # 1) Prefer current working dir (your PowerShell location)
    cwd = Path.cwd()
    if (cwd / REL_PREDS).exists():
        return cwd
    # 2) Fallback: search upwards from script location
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / REL_PREDS).exists():
            return p
    # 3) Last resort: return cwd (error will be raised later if file not found)
    return cwd

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
        # budget left for free set after fixing some at cap
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

def main():
    root = detect_project_root()
    preds_path = root / REL_PREDS
    out_dir    = root / REL_OUT_DIR
    out_csv    = root / REL_OUT_CSV
    out_meta   = root / REL_OUT_META

    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {preds_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    preds = pd.read_csv(preds_path)
    if "month" not in preds.columns:
        raise ValueError("preds_panel.csv must have a 'month' column")
    deciles = [c for c in preds.columns if c.startswith("ME")]
    if len(deciles) != 10:
        raise ValueError(f"Expected 10 ME columns, found: {deciles}")

    rows = []
    months_all_nonpos = 0
    for _, r in preds.iterrows():
        mu = r[deciles].astype(float).values
        pos = np.maximum(mu, 0.0)
        base = np.full_like(pos, 1.0 / len(pos)) if pos.sum() == 0 else (pos / pos.sum())
        if pos.sum() == 0:
            months_all_nonpos += 1
        w = project_capped_simplex(base, WEIGHT_CAP)
        rows.append({"month": r["month"], **{d: w[i] for i, d in enumerate(deciles)}})

    W = pd.DataFrame(rows)
    # Sanity
    if W[deciles].isna().any().any():
        raise AssertionError("NaNs detected in weights.")
    sums = W[deciles].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-12):
        # final safeguard renorm (shouldn't trigger, but harmless)
        W[deciles] = W[deciles].div(W[deciles].sum(axis=1), axis=0)
        sums = W[deciles].sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise AssertionError("Row weights do not sum to 1.0 after renormalization.")

    W.to_csv(out_csv, index=False)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "method": "proportional_to_positive_preds + capped-simplex projection",
            "cap": WEIGHT_CAP,
            "inputs": {"preds_panel": str(preds_path.relative_to(root))},
            "deciles": deciles,
            "notes": f"Months with all non-positive preds → equal-weight: {months_all_nonpos}"
        }, f, indent=2)

    print(f"Saved weights → {out_csv}")
    print(f"Saved meta    → {out_meta}")
    print(f"Months with all non-positive preds: {months_all_nonpos}")

if __name__ == "__main__":
    main()
