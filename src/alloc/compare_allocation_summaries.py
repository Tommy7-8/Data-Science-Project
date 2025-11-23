# src/alloc/compare_allocation_summaries.py
#
# Compare the performance summary of:
#   - Baseline LR mean–variance allocator
#   - GB mean–variance allocator
#
# Inputs (JSON summaries created by evaluation scripts):
#   results/alloc_lr/performance_baseline_meta.json
#   results/alloc_gb/performance_gb_mv_meta.json
#
# Output:
#   results/alloc_comparison/summary_lr_vs_gb.csv
#       aligned semicolon CSV comparing metrics + differences (GB − LR)
#

import json
from pathlib import Path
import pandas as pd


# ---- Paths to JSON summaries ----
BASE_META_PATH = Path("results/alloc_lr/performance_baseline_meta.json")
GB_META_PATH   = Path("results/alloc_gb/performance_gb_mv_meta.json")

# ---- Output paths ----
OUT_DIR     = Path("results/alloc_comparison")
OUT_SUMMARY = OUT_DIR / "summary_lr_vs_gb.csv"


# ---------- number formatting (shared across project) ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """
    Format numbers by trimming unnecessary trailing zeros while keeping
    up to `decimals` digits. Return empty if value is None or NaN.
    """
    if val is None or pd.isna(val):
        return ""
    try:
        s = f"{float(val):.{decimals}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    except Exception:
        # fallback for non-numeric values
        return str(val)


def write_aligned_csv(
    df: pd.DataFrame,
    path: Path,
    decimals: int = 6,
    sep: str = ";",
) -> None:
    """
    Write a DataFrame as a semicolon-separated, visually aligned CSV:
      - 'metric' column left-aligned
      - numeric fields right-aligned
      - trimmed decimal formatting for numeric cells
    """
    df_str = df.copy()

    left_cols = ["metric"]
    right_cols = [c for c in df_str.columns if c not in left_cols]

    # Format right-aligned numeric-looking columns
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

    # Header
    header_cells = [
        f"{c:<{widths[c]}}" if c in left_cols else f"{c:>{widths[c]}}"
        for c in df_str.columns
    ]
    lines = [sep.join(header_cells)]

    # Rows
    for _, row in df_str.iterrows():
        cells = [
            f"{row[c]:<{widths[c]}}" if c in left_cols else f"{row[c]:>{widths[c]}}"
            for c in df_str.columns
        ]
        lines.append(sep.join(cells))

    path.write_text("\n".join(lines), encoding="utf-8-sig")


# ---------- Main comparison pipeline ----------

def main() -> None:
    """
    Compare LR vs GB allocation summaries:

      1. Load baseline LR summary JSON.
      2. Load GB MV summary JSON.
      3. For each metric, record LR value, GB value, and the difference (GB − LR).
      4. Write an aligned semicolon CSV to results/alloc_comparison/.
      5. Print a console diff for quick inspection.
    """
    root = Path.cwd()
    base_meta_p = root / BASE_META_PATH
    gb_meta_p   = root / GB_META_PATH

    # Check existence
    if not base_meta_p.exists():
        raise FileNotFoundError(f"Missing baseline LR meta JSON: {base_meta_p}")
    if not gb_meta_p.exists():
        raise FileNotFoundError(f"Missing GB meta JSON: {gb_meta_p}")

    # Load JSON summaries
    with open(base_meta_p, "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    with open(gb_meta_p, "r", encoding="utf-8") as f:
        gb_meta = json.load(f)

    # Order of metrics for output
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

        # Compute difference when both numbers are present
        diff = None
        try:
            if base_val is not None and gb_val is not None:
                diff = float(gb_val) - float(base_val)
        except Exception:
            diff = None

        rows.append({
            "metric": m,
            "baseline_lr": base_val,
            "gb_mv": gb_val,
            "gb_minus_baseline": diff,
        })

    df = pd.DataFrame(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(df, root / OUT_SUMMARY, decimals=6, sep=";")

    # Console display
    print(f"Saved LR vs GB summary → {OUT_SUMMARY}")
    print("\nSummary (gb_mv - baseline_lr):")
    for _, row in df.iterrows():
        metric = row["metric"]
        value = row["gb_minus_baseline"]
        if value is None or pd.isna(value):
            print(f" {metric:30s}: n/a")
        else:
            print(f" {metric:30s}: {value:.6f}")


if __name__ == "__main__":
    main()
