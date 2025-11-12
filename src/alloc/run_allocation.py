# src/alloc/run_allocation.py
# Baseline allocator — proportional-to-positive preds + capped-simplex projection.
# Reads aligned semicolon CSVs; writes aligned semicolon CSVs.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

REL_PREDS     = Path("results/oos_panel/preds_panel.csv")
REL_OUT_DIR   = Path("results/alloc")
REL_OUT_CSV   = REL_OUT_DIR / "weights_baseline.csv"
REL_OUT_META  = REL_OUT_DIR / "weights_baseline_meta.json"
WEIGHT_CAP    = 0.40  # per-asset cap (feasible for 10 names because 10 * 0.40 >= 1)

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

# ---------- robust loader for aligned semicolon preds_panel.csv ----------
def load_preds_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    # Read with semicolon; strip header and cell whitespace; handle BOM
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", engine="python")
        # normalize headers and cells
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    else:
        # fallback for plain CSV
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    # require month column (after stripping)
    if "month" not in df.columns:
        raise ValueError("preds_panel.csv must have a 'month' column")

    # order decile columns ME1..ME10 and convert to floats
    deciles = [c for c in df.columns if c.startswith("ME")]
    # enforce numeric order
    ordered = [f"ME{i}" for i in range(1, 11) if f"ME{i}" in deciles]
    if len(ordered) != 10:
        raise ValueError(f"Expected 10 ME columns, found: {sorted(deciles)}")

    # convert numeric cells
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

def detect_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / REL_PREDS).exists():
        return cwd
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / REL_PREDS).exists():
            return p
    return cwd

def main():
    root      = detect_project_root()
    preds_p   = root / REL_PREDS
    out_dir   = root / REL_OUT_DIR
    out_csv   = root / REL_OUT_CSV
    out_meta  = root / REL_OUT_META

    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_preds_panel(preds_p)  # robust read + clean + ME1..ME10 order
    deciles = [c for c in preds.columns if c.startswith("ME")]  # already ordered

    rows = []
    months_all_nonpos = 0
    for _, r in preds.iterrows():
        mu = r[deciles].astype(float).values  # predicted returns
        pos = np.maximum(mu, 0.0)             # zero-out negatives
        if pos.sum() == 0:
            base = np.full_like(pos, 1.0 / len(pos))  # equal-weight if all <= 0
            months_all_nonpos += 1
        else:
            base = pos / pos.sum()

        w = project_capped_simplex(base, WEIGHT_CAP)
        rows.append({"month": r["month"], **{d: w[i] for i, d in enumerate(deciles)}})

    W = pd.DataFrame(rows, columns=["month"] + deciles)

    # Sanity: row sums ~ 1
    if W[deciles].isna().any().any():
        raise AssertionError("NaNs detected in weights.")
    sums = W[deciles].sum(axis=1).to_numpy()
    if not np.allclose(sums, 1.0, atol=1e-12):
        W[deciles] = W[deciles].div(W[deciles].sum(axis=1), axis=0)
        sums = W[deciles].sum(axis=1).to_numpy()
        if not np.allclose(sums, 1.0, atol=1e-9):
            raise AssertionError("Row weights do not sum to 1.0 after renormalization.")

    # Write aligned CSV + meta
    write_aligned_csv(W, out_csv, decimals=6, sep=";")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump({
            "method": "proportional_to_positive_preds + capped-simplex projection",
            "cap": WEIGHT_CAP,
            "inputs": {"preds_panel": str(preds_p.relative_to(root))},
            "deciles": deciles,
            "notes": f"Months with all non-positive preds → equal-weight: {months_all_nonpos}"
        }, f, indent=2)

    print(f"Saved weights → {out_csv}")
    print(f"Saved meta    → {out_meta}")
    print(f"Months with all non-positive preds: {months_all_nonpos}")

if __name__ == "__main__":
    main()
