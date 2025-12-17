# src/alloc/evaluate_allocation_lr.py
# Evaluate realized portfolio performance for LR weights using true returns.
# Includes transaction costs.
# Note: the turnover cap is applied only inside the transaction-cost penalty
# (it does not constrain the portfolio weights).
#
# Inputs:
#   - results/alloc_lr/weights_baseline.csv      (LR allocation weights, ME1..ME10)
#   - results/oos_panel_lr/true_panel_lr.csv     (realized ME1..ME10 returns)
#
# Outputs:
#   - results/alloc_lr/performance_baseline.csv       (monthly performance, aligned ;)
#   - results/alloc_lr/performance_baseline_meta.json (summary statistics)

import json
from pathlib import Path

import numpy as np
import pandas as pd

# LR-specific paths (relative to project root / current working directory)
REL_WEIGHTS = Path("results/alloc_lr/weights_baseline.csv")
REL_TRUES = Path("results/oos_panel_lr/true_panel_lr.csv")

REL_OUT_DIR = Path("results/alloc_lr")
REL_OUT_CSV = REL_OUT_DIR / "performance_baseline.csv"
REL_OUT_META = REL_OUT_DIR / "performance_baseline_meta.json"

# ---- Parameters for realism ----
TRANSACTION_COST = 0.001   # 0.10% per 100% turnover (10 bps per full notional traded)
TURNOVER_LIMIT = 0.20      # max 20% turnover per month


# ---------- pretty CSV writer (semicolon, aligned, trimmed zeros) ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """
    Format a number with up to `decimals` decimal places,
    removing unnecessary trailing zeros and decimal points.

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
      - numeric-like entries formatted via _fmt_num
    """
    df_str = df.copy()

    # 'month' is left-aligned, everything else right-aligned
    left_cols = [c for c in df_str.columns if c == "month"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format numeric-looking columns on the right
    for c in right_cols:
        ser = pd.to_numeric(df_str[c], errors="coerce")
        if ser.notna().any():
            df_str[c] = ser.map(
                lambda v: _fmt_num(v, decimals) if pd.notna(v) else ""
            )
        else:
            df_str[c] = df_str[c].astype(str)

    # Convert to strings for width computation
    df_str = df_str.astype(str)
    widths = {
        c: max(len(c), df_str[c].map(len).max())
        for c in df_str.columns
    }

    # Build header line
    header = [
        f"{c:<{widths[c]}}" if c in left_cols else f"{c:>{widths[c]}}"
        for c in df_str.columns
    ]
    lines = [sep.join(header)]

    # Build data lines
    for _, row in df_str.iterrows():
        cells = [
            f"{row[c]:<{widths[c]}}" if c in left_cols else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        lines.append(sep.join(cells))

    Path(path).write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- performance helpers ----------

def compute_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown of a return series.

    Args:
        returns: monthly portfolio returns (net or gross).

    Returns:
        Minimum value of the cumulative return path relative to previous peaks
        (e.g. -0.30 for a 30% max drawdown).
    """
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1.0
    return float(dd.min())


def compute_turnover(weights_df: pd.DataFrame, deciles: list[str]) -> pd.Series:
    """
    Compute monthly portfolio turnover, given a panel of weights.

    Definition:
        turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|

    Args:
        weights_df: DataFrame with columns ME1..ME10 (and possibly 'month').
        deciles: list of decile column names, e.g. ["ME1", ..., "ME10"].

    Returns:
        Series with monthly turnover (NaN for the first month).
    """
    w_prev = weights_df[deciles].shift(1)
    turnover = (weights_df[deciles] - w_prev).abs().sum(axis=1) / 2.0
    return turnover


def main() -> None:
    """
    Evaluate the LR portfolio allocation using realized returns.

    Steps:
      1. Load LR weights and true ME returns (semicolon-aligned CSVs).
      2. Align them by month and compute:
           - gross monthly portfolio returns
           - turnover and turnover after applying the cap (for transaction-cost penalty)
           - net returns after transaction costs
      3. Compute summary statistics:
           - mean monthly return, annualized return/vol, Sharpe, max drawdown
           - average turnover before and after the turnover cap (cost penalty)
      4. Save monthly performance CSV and JSON summary.
    """
    root = Path.cwd()

    weights_p = root / REL_WEIGHTS
    trues_p = root / REL_TRUES
    out_dir = root / REL_OUT_DIR
    out_csv = root / REL_OUT_CSV
    out_meta = root / REL_OUT_META

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load weights and true returns (semicolon-separated, aligned) ----
    # Both CSVs are produced by our own pipeline, so we can assume ';' as separator.
    W = pd.read_csv(weights_p, sep=";", encoding="utf-8-sig")
    R = pd.read_csv(trues_p, sep=";", encoding="utf-8-sig")

    # Clean up column names (strip spaces / BOM)
    W.columns = [c.strip() for c in W.columns]
    R.columns = [c.strip() for c in R.columns]

    if "month" not in W.columns or "month" not in R.columns:
        raise ValueError(
            "Both weights_baseline.csv and true_panel_lr.csv must have a 'month' column."
        )

    # Identify the ME decile columns in the weights file
    deciles = [c for c in W.columns if c.startswith("ME")]
    if len(deciles) != 10:
        raise ValueError(
            f"Expected 10 ME columns in weights file, found {deciles}"
        )

    # ---- Align on common months ----
    # Suffixes keep weights and returns separate after merge.
    df = pd.merge(W, R, on="month", suffixes=("_w", "_r"))
    df = df.sort_values("month").reset_index(drop=True)

    # ---- Compute gross portfolio return each month ----
    # ret_cols: realized returns per ME decile
    # w_cols:   portfolio weights per ME decile
    ret_cols = [f"{d}_r" for d in deciles]
    w_cols = [f"{d}_w" for d in deciles]

    # Element-wise multiply then sum across deciles
    df["gross_ret"] = (df[w_cols].values * df[ret_cols].values).sum(axis=1)

    # ---- Compute turnover and apply turnover cap (for transaction-cost penalty only) ----
    # Build a clean weights-only DF with ME1..ME10 columns for turnover calculation
    W_clean = df[w_cols].rename(columns={f"{d}_w": d for d in deciles})
    W_clean["month"] = df["month"]

    df["turnover"] = compute_turnover(W_clean, deciles)
    df["turnover_limited"] = df["turnover"].clip(upper=TURNOVER_LIMIT)

    # ---- Apply transaction cost penalty ----
    # Cost = turnover_after_limit * TRANSACTION_COST
    df["net_ret"] = df["gross_ret"] - df["turnover_limited"] * TRANSACTION_COST

    # ---- Compute summary statistics ----
    rets = df["net_ret"]

    mean_ret = float(rets.mean())
    vol = float(rets.std())
    sharpe = (mean_ret / vol * np.sqrt(12)) if vol > 0 else np.nan
    ann_ret = (1.0 + mean_ret) ** 12 - 1.0
    ann_vol = vol * np.sqrt(12)
    dd = compute_drawdown(rets)

    avg_turn = float(df["turnover"].mean())
    avg_turn_limited = float(df["turnover_limited"].mean())

    summary = {
        "mean_monthly_return": mean_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd,
        "avg_turnover": avg_turn,
        "avg_turnover_after_limit": avg_turn_limited,
        "transaction_cost": TRANSACTION_COST,
        "turnover_limit": TURNOVER_LIMIT,
    }

    # ---- Save aligned CSV of monthly performance ----
    out_df = df[["month", "gross_ret", "net_ret", "turnover", "turnover_limited"]].copy()
    write_aligned_csv(out_df, out_csv, decimals=6, sep=";")

    # ---- Save JSON summary ----
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Console summary ----
    print(f"Saved LR monthly performance → {out_csv}")
    print(f"Saved LR summary JSON        → {out_meta}")
    print("\nKey LR metrics:")
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            print(f" {k:30s}: {v:.6f}")
        else:
            print(f" {k:30s}: {v}")


if __name__ == "__main__":
    main()
