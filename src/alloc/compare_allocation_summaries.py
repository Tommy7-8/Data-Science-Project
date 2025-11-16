# src/alloc/compare_allocation_summaries.py
# Compare performance of:
#   - Baseline (linear regression MV allocator)
#   - GB (gradient boosting MV allocator)
#
# Reads:
#   - results/alloc/performance_baseline_meta.json
#   - results/alloc_gb/performance_gb_mv_meta.json
#
# Writes:
#   - results/alloc_comparison/summary_lr_vs_gb.csv  (semicolon, aligned)
# and prints a small console summary.

import json
from pathlib import Path
import pandas as pd

BASE_META_PATH = Path("results/alloc/performance_baseline_meta.json")
GB_META_PATH   = Path("results/alloc_gb/performance_gb_mv_meta.json")

OUT_DIR        = Path("results/alloc_comparison")
OUT_SUMMARY    = OUT_DIR / "summary_lr_vs_gb.csv"


def _fmt_num(val, decimals=6):
    if val is None:
        return ""
    try:
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def write_aligned_csv(df: pd.DataFrame, path: Path, decimals: int = 6, sep: str = ";"):
    df_str = df.copy()
    left_cols = [c for c in df_str.columns if c == "metric"]
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

    path.write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    root = Path.cwd()
    base_meta_p = root / BASE_META_PATH
    gb_meta_p   = root / GB_META_PATH

    if not base_meta_p.exists():
        raise FileNotFoundError(f"Missing baseline meta JSON: {base_meta_p}")
    if not gb_meta_p.exists():
        raise FileNotFoundError(f"Missing GB meta JSON: {gb_meta_p}")

    with open(base_meta_p, "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    with open(gb_meta_p, "r", encoding="utf-8") as f:
        gb_meta = json.load(f)

    # pick a consistent set of metrics
    metrics_order = [
        "mean_monthly_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "avg_turnover",
        "avg_turnover_after_limit",
        "transaction_cost",
        "turnover_limit",
    ]

    rows = []
    for m in metrics_order:
        base_val = base_meta.get(m, None)
        gb_val   = gb_meta.get(m, None)
        diff     = None
        try:
            if (base_val is not None) and (gb_val is not None):
                diff = float(gb_val) - float(base_val)
        except Exception:
            diff = None

        rows.append(
            {
                "metric": m,
                "baseline_lr": base_val,
                "gb_mv": gb_val,
                "gb_minus_baseline": diff,
            }
        )

    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(df, root / OUT_SUMMARY, decimals=6, sep=";")

    print(f"Saved LR vs GB summary → {OUT_SUMMARY}")
    print("\nSummary (gb_mv - baseline_lr):")
    for _, row in df.iterrows():
        m = row["metric"]
        d = row["gb_minus_baseline"]
        if d is None or pd.isna(d):
            print(f" {m:30s}: n/a")
        else:
            print(f" {m:30s}: {d:.6f}")


if __name__ == "__main__":
    main()
