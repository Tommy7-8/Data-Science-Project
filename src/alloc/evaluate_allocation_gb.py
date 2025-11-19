# src/alloc/evaluate_allocation_gb_mv.py
# Evaluate realized portfolio performance for GB weights using true LR returns.
# Includes transaction costs and turnover limits.
# Outputs aligned CSV (semicolon, trimmed decimals) and JSON summary.

import json
from pathlib import Path
import numpy as np
import pandas as pd

# --- GB-specific paths ---
REL_WEIGHTS  = Path("results/alloc_gb/weights_gb_mv.csv")
REL_TRUES    = Path("results/oos_panel_lr/true_panel_lr.csv")   # <-- FIXED
REL_OUT_DIR  = Path("results/alloc_gb")
REL_OUT_CSV  = REL_OUT_DIR / "performance_gb_mv.csv"
REL_OUT_META = REL_OUT_DIR / "performance_gb_mv_meta.json"

# ---- Parameters ----
TRANSACTION_COST = 0.001
TURNOVER_LIMIT   = 0.20

# ---------- Trimmed, aligned CSV writer ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    s = f"{float(val):.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s

def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";"):
    df_str = df.copy()
    left_cols  = ["month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(lambda v: _fmt_num(v, decimals) if pd.notna(v) else "")
        else:
            df_str[c] = df_str[c].astype(str)

    df_str = df_str.astype(str)
    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    header = [
        f"{c:<{widths[c]}}" if c in left_cols else f"{c:>{widths[c]}}"
        for c in df_str.columns
    ]
    lines = [";".join(header)]

    for _, row in df_str.iterrows():
        cells = [
            f"{row[c]:<{widths[c]}}" if c in left_cols else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        lines.append(";".join(cells))

    path.write_text("\n".join(lines), encoding="utf-8-sig")

# ---------- Helpers ----------
def compute_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())

def compute_turnover(weights_df: pd.DataFrame, deciles) -> pd.Series:
    w_prev = weights_df[deciles].shift(1)
    return (weights_df[deciles] - w_prev).abs().sum(axis=1) / 2.0

# ---------- MAIN ----------
def main():
    root = Path.cwd()
    weights_p = root / REL_WEIGHTS
    trues_p   = root / REL_TRUES
    out_dir   = root / REL_OUT_DIR
    out_csv   = root / REL_OUT_CSV
    out_meta  = root / REL_OUT_META
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load weights + true returns ---
    W = pd.read_csv(weights_p, sep=";", encoding="utf-8-sig")
    R = pd.read_csv(trues_p,  sep=";", encoding="utf-8-sig")

    W.columns = [c.strip() for c in W.columns]
    R.columns = [c.strip() for c in R.columns]

    if "month" not in W.columns or "month" not in R.columns:
        raise ValueError("Both weights_gb_mv.csv and true_panel_lr.csv must contain 'month'.")

    deciles = [c for c in W.columns if c.startswith("ME")]
    if len(deciles) != 10:
        raise ValueError(f"Expected 10 ME columns in weights, found: {deciles}")

    # --- Merge and align ---
    df = pd.merge(W, R, on="month", suffixes=("_w", "_r"))
    df = df.sort_values("month").reset_index(drop=True)

    # --- Compute gross returns ---
    w_cols   = [f"{d}_w" for d in deciles]
    ret_cols = [f"{d}_r" for d in deciles]
    df["gross_ret"] = (df[w_cols].values * df[ret_cols].values).sum(axis=1)

    # --- Turnover ---
    W_turn = df[w_cols].copy()
    W_turn.columns = [c.replace("_w", "") for c in W_turn.columns]
    W_turn["month"] = df["month"]

    df["turnover"] = compute_turnover(W_turn, deciles)
    df["turnover_limited"] = df["turnover"].clip(upper=TURNOVER_LIMIT)

    # --- Net returns ---
    df["net_ret"] = df["gross_ret"] - df["turnover_limited"] * TRANSACTION_COST

    # --- Summary statistics ---
    rets = df["net_ret"]
    mean_ret = float(rets.mean())
    vol      = float(rets.std())
    sharpe   = (mean_ret / vol * np.sqrt(12)) if vol > 0 else np.nan
    ann_ret  = (1 + mean_ret) ** 12 - 1
    ann_vol  = vol * np.sqrt(12)
    dd       = compute_drawdown(rets)

    summary = {
        "mean_monthly_return": mean_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "avg_turnover": float(df["turnover"].mean()),
        "avg_turnover_after_limit": float(df["turnover_limited"].mean()),
        "transaction_cost": TRANSACTION_COST,
        "turnover_limit": TURNOVER_LIMIT,
    }

    # --- Save results ---
    out_df = df[["month", "gross_ret", "net_ret", "turnover", "turnover_limited"]]
    write_aligned_csv(out_df, out_csv, decimals=6, sep=";")

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved GB monthly performance → {out_csv}")
    print(f"Saved GB summary JSON        → {out_meta}")

    print("\nGB MV key metrics:")
    for k, v in summary.items():
        print(f" {k:30s}: {v if not isinstance(v, float) else f'{v:.6f}'}")

if __name__ == "__main__":
    main()
