# src/alloc/evaluate_allocation_gb.py
# Evaluate realized portfolio performance for GB-based weights using true realized returns.
#
# This script mirrors the LR evaluation logic, but for the GB mean–variance allocator.
# Note: the turnover cap is applied only inside the transaction-cost penalty
# (it does not constrain the portfolio weights).
#
# Inputs:
#   - results/alloc_gb/weights_gb_mv.csv         (GB allocation weights, ME1..ME10)
#   - results/oos_panel_lr/true_panel_lr.csv     (realized ME1..ME10 returns)
#
# Outputs:
#   - results/alloc_gb/performance_gb_mv.csv         (monthly performance, aligned ;)
#   - results/alloc_gb/performance_gb_mv_meta.json   (summary statistics)

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- GB-specific paths (relative to project root) ---
REL_WEIGHTS = Path("results/alloc_gb/weights_gb_mv.csv")
REL_TRUES = Path("results/oos_panel_lr/true_panel_lr.csv")   # realized returns (LR panel)
REL_OUT_DIR = Path("results/alloc_gb")
REL_OUT_CSV = REL_OUT_DIR / "performance_gb_mv.csv"
REL_OUT_META = REL_OUT_DIR / "performance_gb_mv_meta.json"

# ---- Parameters ----
TRANSACTION_COST = 0.001   # cost per 1.0 turnover (10 bps)
TURNOVER_LIMIT = 0.20      # cap turnover at 20% per month


# ---------- Trimmed, aligned CSV writer ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """
    Format a number with at most `decimals` decimal places,
    removing trailing zeros and trailing decimal points.

    Returns empty string for NaN.
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
    Write DataFrame as a semicolon-separated, visually aligned CSV.

    Conventions:
      - 'month' column is left-aligned
      - all other columns are right-aligned
      - numeric values formatted via _fmt_num
    """
    df_str = df.copy()

    left_cols = ["month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-like columns
    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(
                lambda v: _fmt_num(v, decimals) if pd.notna(v) else ""
            )
        else:
            df_str[c] = df_str[c].astype(str)

    df_str = df_str.astype(str)
    widths = {
        c: max(len(c), df_str[c].map(len).max())
        for c in df_str.columns
    }

    # Header row
    header = [
        f"{c:<{widths[c]}}" if c in left_cols else f"{c:>{widths[c]}}"
        for c in df_str.columns
    ]
    lines = [";".join(header)]

    # Data rows
    for _, row in df_str.iterrows():
        cells = [
            f"{row[c]:<{widths[c]}}" if c in left_cols else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        lines.append(";".join(cells))

    path.write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- Helpers ----------

def compute_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown from a series of returns.

    Returns:
        Minimum drawdown value (e.g., -0.30 for a 30% max drawdown).
    """
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def compute_turnover(weights_df: pd.DataFrame, deciles: list[str]) -> pd.Series:
    """
    Compute monthly turnover given a panel of weights.

    Turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|

    Args:
        weights_df: DataFrame with at least columns ME1..ME10 (and 'month').
        deciles: list of ME column names.

    Returns:
        Series of turnover values per month (NaN for the first month).
    """
    w_prev = weights_df[deciles].shift(1)
    return (weights_df[deciles] - w_prev).abs().sum(axis=1) / 2.0


# ---------- MAIN ----------

def main() -> None:
    """
    Evaluate the GB mean–variance allocation:

      1. Load GB weights and LR true returns (semicolon-aligned CSVs).
      2. Align both on 'month'.
      3. Compute:
           - gross portfolio return
           - turnover and turnover_limited (cap only affects the cost penalty)
           - net returns after transaction costs
      4. Derive summary performance statistics:
           - mean monthly return, annualized return/vol, Sharpe, max drawdown
           - average turnover (before and after limit)
      5. Save monthly performance CSV and JSON summary.
    """
    root = Path.cwd()
    weights_p = root / REL_WEIGHTS
    trues_p = root / REL_TRUES
    out_dir = root / REL_OUT_DIR
    out_csv = root / REL_OUT_CSV
    out_meta = root / REL_OUT_META
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load weights + true returns (semicolon-separated) ---
    W = pd.read_csv(weights_p, sep=";", encoding="utf-8-sig")
    R = pd.read_csv(trues_p, sep=";", encoding="utf-8-sig")

    # Normalize column names
    W.columns = [c.strip() for c in W.columns]
    R.columns = [c.strip() for c in R.columns]

    if "month" not in W.columns or "month" not in R.columns:
        raise ValueError(
            "Both weights_gb_mv.csv and true_panel_lr.csv must contain a 'month' column."
        )

    deciles = [c for c in W.columns if c.startswith("ME")]
    if len(deciles) != 10:
        raise ValueError(f"Expected 10 ME columns in weights, found: {deciles}")

    # --- Merge and align by month ---
    df = pd.merge(W, R, on="month", suffixes=("_w", "_r"))
    df = df.sort_values("month").reset_index(drop=True)

    # --- Compute gross portfolio returns ---
    w_cols = [f"{d}_w" for d in deciles]
    ret_cols = [f"{d}_r" for d in deciles]
    df["gross_ret"] = (df[w_cols].values * df[ret_cols].values).sum(axis=1)

    # --- Turnover computation (in ME1..ME10 space) ---
    W_turn = df[w_cols].copy()
    W_turn.columns = [c.replace("_w", "") for c in W_turn.columns]  # back to ME1..ME10
    W_turn["month"] = df["month"]

    df["turnover"] = compute_turnover(W_turn, deciles)
    df["turnover_limited"] = df["turnover"].clip(upper=TURNOVER_LIMIT)

    # --- Net returns after transaction costs ---
    df["net_ret"] = df["gross_ret"] - df["turnover_limited"] * TRANSACTION_COST

    # --- Summary statistics ---
    rets = df["net_ret"]
    mean_ret = float(rets.mean())
    vol = float(rets.std())
    sharpe = (mean_ret / vol * np.sqrt(12)) if vol > 0 else np.nan
    ann_ret = (1 + mean_ret) ** 12 - 1
    ann_vol = vol * np.sqrt(12)
    dd = compute_drawdown(rets)

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

    # --- Save monthly performance series ---
    out_df = df[["month", "gross_ret", "net_ret", "turnover", "turnover_limited"]].copy()
    write_aligned_csv(out_df, out_csv, decimals=6, sep=";")

    # --- Save JSON summary ---
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- Console summary ---
    print(f"Saved GB monthly performance → {out_csv}")
    print(f"Saved GB summary JSON        → {out_meta}")
    print("\nGB MV key metrics:")
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            print(f" {k:30s}: {v:.6f}")
        else:
            print(f" {k:30s}: {v}")


if __name__ == "__main__":
    main()
