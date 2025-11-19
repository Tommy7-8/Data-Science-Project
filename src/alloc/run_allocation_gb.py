# src/alloc/run_allocation_gb.py
# GB mean-variance allocator using GB predicted returns + vol predictions.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- paths ----------
REL_PREDS_RET  = Path("results/oos_panel_gb/preds_panel_gb.csv")
REL_PREDS_VOL  = Path("results/oos_vol_panel_gb/preds_vol_panel_gb.csv")
REL_TRUE_RET   = Path("results/oos_panel_lr/true_panel_lr.csv")  # realized returns

REL_OUT_DIR    = Path("results/alloc_gb")
REL_OUT_CSV    = REL_OUT_DIR / "weights_gb_mv.csv"
REL_OUT_META   = REL_OUT_DIR / "weights_gb_mv_meta.json"

# ---------- hyperparameters ----------
WEIGHT_CAP  = 0.40
COV_WINDOW  = 120
MIN_COV_OBS = 24
SHRINKAGE   = 0.50
VOL_BLEND   = 0.50
EPS_VAR     = 1e-6


# ---------- aligned CSV writer ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    s = f"{float(val):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def write_aligned_csv(df: pd.DataFrame, path: Path, decimals=6, sep=";"):
    df_str = df.copy()
    left_cols = ["month"]
    right_cols = [c for c in df_str.columns if c != "month"]

    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        df_str[c] = ser.map(lambda v: _fmt_num(v, decimals))  

    df_str = df_str.astype(str)
    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    header = [f"{c:<{widths[c]}}" if c=="month" else f"{c:>{widths[c]}}" for c in df_str.columns]
    out = [sep.join(header)]

    for _, row in df_str.iterrows():
        line = [
            f"{row[c]:<{widths[c]}}" if c=="month" else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        out.append(sep.join(line))

    path.write_text("\n".join(out), encoding="utf-8-sig")


# ---------- loader (semicolon or normal) ----------
def load_panel(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        head = f.readline()

    if ";" in head:
        df = pd.read_csv(path, sep=";", dtype=str, engine="python")
        df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
        obj = df.select_dtypes(include="object").columns
        df[obj] = df[obj].apply(lambda s: s.str.strip())
    else:
        df = pd.read_csv(path, dtype=str)

    if "month" not in df.columns:
        raise ValueError(f"{path} missing 'month' column")

    deciles = [f"ME{i}" for i in range(1, 11)]
    for d in deciles:
        df[d] = pd.to_numeric(df[d], errors="coerce")

    return df[["month"] + deciles]


# ---------- projection ----------
def project_capped_simplex(w0, cap):
    w0 = np.clip(w0, 0, None)
    n = len(w0)
    if n*cap < 1:
        cap = 1/n

    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(n, 1/n)

    fixed = np.zeros(n, bool)
    w = np.zeros(n)

    while True:
        free = ~fixed
        budget = 1 - fixed.sum()*cap
        if free.sum()==0:
            w[fixed]=cap
            return w/w.sum()

        denom = w0[free].sum()
        w[free] = (w0[free]/denom)*budget if denom>0 else budget/free.sum()
        w[fixed] = cap

        over = (w>cap)&free
        if not over.any():
            return w/w.sum()
        fixed[over]=True
        w[over]=0


# ---------- covariance ----------
def build_covariance(hist, pred_var):
    n = len(pred_var)

    if len(hist)<MIN_COV_OBS:
        return np.diag(np.maximum(pred_var, EPS_VAR))

    S = np.cov(hist, rowvar=False, ddof=1)
    if S.shape!=(n,n):
        return np.diag(np.maximum(pred_var,EPS_VAR))

    diagS = np.diag(np.diag(S))
    Sigma = (1-SHRINKAGE)*S + SHRINKAGE*diagS

    diagSigma = np.diag(Sigma)
    blended = VOL_BLEND*diagSigma + (1-VOL_BLEND)*pred_var
    np.fill_diagonal(Sigma, blended)
    return Sigma


# ---------- main ----------
def main():
    root = Path.cwd()

    preds_ret = load_panel(root/REL_PREDS_RET)
    preds_vol = load_panel(root/REL_PREDS_VOL)
    true_ret  = load_panel(root/REL_TRUE_RET)

    dec = [f"ME{i}" for i in range(1,11)]
    preds_ret = preds_ret.rename(columns={d:f"{d}_ret" for d in dec})
    preds_vol = preds_vol.rename(columns={d:f"{d}_vol" for d in dec})
    true_ret  = true_ret.rename(columns={d:f"{d}_true" for d in dec})

    df = preds_ret.merge(preds_vol,on="month").merge(true_ret,on="month")
    df = df.sort_values("month").reset_index(drop=True)

    out = []
    for idx,row in df.iterrows():
        mu = np.array([row[f"{d}_ret"] for d in dec], float)
        pred_var = np.array([row[f"{d}_vol"] for d in dec], float)

        start = max(0,idx-COV_WINDOW)
        hist = df.iloc[start:idx][[f"{d}_true" for d in dec]].to_numpy()

        Sigma = build_covariance(hist, pred_var)
        w_dir = np.linalg.solve(Sigma, mu) if np.linalg.det(Sigma)!=0 else mu
        w = project_capped_simplex(w_dir, WEIGHT_CAP)

        out.append({"month":row["month"], **{d:w[i] for i,d in enumerate(dec)}})

    W = pd.DataFrame(out)

    REL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(W, root/REL_OUT_CSV)

    meta = {
        "method":"GB mean-variance",
        "weight_cap":WEIGHT_CAP,
        "inputs":{
            "gb_pred_returns": str(REL_PREDS_RET),
            "gb_pred_vol": str(REL_PREDS_VOL),
            "true_returns": str(REL_TRUE_RET)
        }
    }
    (root/REL_OUT_META).write_text(json.dumps(meta,indent=2),"utf-8")

    print("Saved GB weights →", REL_OUT_CSV)
    print("Saved GB meta    →", REL_OUT_META)


if __name__=="__main__":
    main()
