# src/alloc/evaluate_allocation.py
# Evaluate realized portfolio performance using weights and true returns.
# Includes transaction costs and turnover limits for realism.
# Outputs aligned CSV (semicolon, dot decimals) and JSON summary.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd

REL_WEIGHTS = Path("results/alloc/weights_baseline.csv")
REL_TRUES   = Path("results/oos_panel/true_panel.csv")
REL_OUT_DIR = Path("results/alloc")
REL_OUT_CSV = REL_OUT_DIR / "performance_baseline.csv"
REL_OUT_META = REL_OUT_DIR / "performance_baseline_meta.json"

# ---- Parameters for realism ----
TRANSACTION_COST = 0.001   # 0.10% per 1.0 turnover (10 bps)
TURNOVER_LIMIT   = 0.20    # max 20% turnover per month

# ---------- pretty CSV writer (semicolon, dot decimals, aligned columns) ----------
def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    return f"{float(val):.{decimals}f}"

def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";"):
    df_str = df.copy()
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(lambda v: _fmt_num(v, decimals) if pd.notna(v) else "")
        else:
            df_str[c] = df_str[c].astype(str)

    df_str = df_str.astype(str)
    widths = {c: max(len(c), df_str[c].map(len).max()) for c in df_str.columns}

    header = [f"{c:<{widths[c]}}" if c in left_cols else f"{c:>{widths[c]}}" for c in df_str.columns]
    lines = [";".join(header)]

    for _, row in df_str.iterrows():
        cells = [f"{row[c]:<{widths[c]}}" if c in left_cols else f"{row[c]:>{widths[c]}}" for c in df_str.columns]
        lines.append(";".join(cells))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")

# ---------- performance helpers ----------
def compute_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return dd.min()

def compute_turnover(weights_df, deciles):
    w_prev = weights_df[deciles].shift(1)
    turnover = (weights_df[deciles] - w_prev).abs().sum(axis=1) / 2.0
    return turnover

def main():
    root = Path.cwd()
    weights_p = root / REL_WEIGHTS
    trues_p   = root / REL_TRUES
    out_dir   = root / REL_OUT_DIR
    out_csv   = root / REL_OUT_CSV
    out_meta  = root / REL_OUT_META
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load weights and true returns (semicolon-separated, aligned) ----
    W = pd.read_csv(weights_p, sep=";", encoding="utf-8-sig")
    R = pd.read_csv(trues_p, sep=";", encoding="utf-8-sig")

    W.columns = [c.strip() for c in W.columns]
    R.columns = [c.strip() for c in R.columns]

    if "month" not in W.columns or "month" not in R.columns:
        raise ValueError("Both weights_baseline.csv and true_panel.csv must have a 'month' column.")

    deciles = [c for c in W.columns if c.startswith("ME")]
    if len(deciles) != 10:
        raise ValueError(f"Expected 10 ME columns in weights file, found {deciles}")

    # ---- Align on common months ----
    df = pd.merge(W, R, on="month", suffixes=("_w", "_r"))
    df = df.sort_values("month").reset_index(drop=True)

    # ---- Compute gross portfolio return each month ----
    ret_cols = [f"{d}_r" for d in deciles]
    w_cols   = [f"{d}_w" for d in deciles]
    df["gross_ret"] = (df[w_cols].values * df[ret_cols].values).sum(axis=1)

    # ---- Compute turnover and apply limits ----
    df["turnover"] = compute_turnover(df[[*w_cols, "month"] + []].rename(columns=lambda c: c.replace("_w", "")), deciles)
    df["turnover_limited"] = df["turnover"].clip(upper=TURNOVER_LIMIT)

    # ---- Apply transaction cost penalty ----
    df["net_ret"] = df["gross_ret"] - df["turnover_limited"] * TRANSACTION_COST

    # ---- Compute summary statistics ----
    rets = df["net_ret"]
    mean_ret = rets.mean()
    vol = rets.std()
    sharpe = (mean_ret / vol * np.sqrt(12)) if vol > 0 else np.nan
    ann_ret = (1 + mean_ret) ** 12 - 1
    ann_vol = vol * np.sqrt(12)
    dd = compute_drawdown(rets)
    avg_turn = df["turnover"].mean()
    avg_turn_limited = df["turnover_limited"].mean()

    summary = {
        "mean_monthly_return": mean_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "avg_turnover": avg_turn,
        "avg_turnover_after_limit": avg_turn_limited,
        "transaction_cost": TRANSACTION_COST,
        "turnover_limit": TURNOVER_LIMIT
    }

    # ---- Save aligned CSV of monthly performance ----
    out_df = df[["month", "gross_ret", "net_ret", "turnover", "turnover_limited"]].copy()
    write_aligned_csv(out_df, out_csv, decimals=6, sep=";")

    # ---- Save JSON summary ----
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Console summary ----
    print(f"Saved monthly performance → {out_csv}")
    print(f"Saved summary JSON        → {out_meta}")
    print("\nKey metrics:")
    for k, v in summary.items():
        print(f" {k:30s}: {v:.6f}")

if __name__ == "__main__":
    main()
