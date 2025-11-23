# src/benchmark/compare_benchmarks.py
#
# Unified benchmark comparison for allocation strategies:
#   - Baseline LR portfolio (net of costs)
#   - GB portfolio (net of costs)
#   - EW_10 (equal-weight of the 10 size deciles)
#   - Market (if available from FF factors)
#
# Reads:
#   - results/alloc_lr/performance_baseline.csv
#   - results/alloc_gb/performance_gb_mv.csv
#   - results/oos_panel_lr/true_panel_lr.csv
#   - data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv
#
# Writes (into results/benchmark/):
#   - benchmarks_panel.csv   : monthly returns of all benchmarks
#   - benchmarks_summary.csv : Sharpe/Sortino with bootstrap CIs, max DD, etc.
#   - benchmarks_tests.csv   : Diebold–Mariano tests vs EW_10 / Market / LR
#

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---- Paths (aligned with project pipeline) ----
PERF_LR_PATH = Path("results/alloc_lr/performance_baseline.csv")
PERF_GB_PATH = Path("results/alloc_gb/performance_gb_mv.csv")

TRUE_PANEL = Path("results/oos_panel_lr/true_panel_lr.csv")
PROC_FULL = Path("data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv")

OUT_DIR = Path("results/benchmark")
PANEL_OUT = OUT_DIR / "benchmarks_panel.csv"
SUMM_OUT = OUT_DIR / "benchmarks_summary.csv"
TESTS_OUT = OUT_DIR / "benchmarks_tests.csv"


# ---------- Formatting helpers ----------

def _fmt_num(val, decimals: int = 6) -> str:
    """
    Format numbers with up to `decimals` decimal places and
    trim unnecessary trailing zeros and decimal points.

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
    Write DataFrame as a semicolon-separated, visually aligned CSV.

      - 'month' column is left-aligned.
      - all other columns are right-aligned.
      - numeric cells are formatted via _fmt_num.

    This keeps the output consistent with the aligned CSVs used in the pipeline.
    """
    df_str = df.copy()

    # Format all columns except 'month' as numbers when possible
    for c in df_str.columns:
        if c != "month":
            ser = pd.to_numeric(df_str[c], errors="coerce")
            if ser.notna().any():
                df_str[c] = ser.map(lambda v: _fmt_num(v, decimals))

    # Compute column widths
    widths = {
        c: max(len(c), df_str[c].astype(str).map(len).max())
        for c in df_str.columns
    }

    # Header row
    header = sep.join(
        f"{c:<{widths[c]}}" if c == "month" else f"{c:>{widths[c]}}"
        for c in df_str.columns
    )
    lines = [header]

    # Data rows
    for _, r in df_str.iterrows():
        line = sep.join(
            f"{r[c]:<{widths[c]}}" if c == "month" else f"{r[c]:>{widths[c]}}"
            for c in df_str.columns
        )
        lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def load_semicolon_csv(path: Path) -> pd.DataFrame:
    """
    Robust loader for our aligned semicolon CSVs.

    - Uses sep=";" and engine="python".
    - Strips BOM and whitespace from headers.
    - Strips whitespace from all string cells.
    """
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df


# ---------- Metric helpers ----------

def max_drawdown(r: pd.Series) -> float:
    """
    Compute maximum drawdown of a return series.

    Args:
        r: monthly returns.

    Returns:
        Minimum of (cum / cummax - 1), e.g. -0.30 for a 30% max DD.
    """
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def sharpe_ci_annualized(
    r: pd.Series,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for annualized Sharpe ratio.

    Args:
        r: monthly returns (Series)
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g. 0.95 for 95% CI)

    Returns:
        (low, high) bounds for annualized Sharpe.
    """
    r = r.dropna().to_numpy()
    if len(r) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(7)
    boot = []

    for _ in range(n_boot):
        s = rng.choice(r, size=len(r), replace=True)
        std = s.std()
        if std > 0:
            boot.append((s.mean() / std) * np.sqrt(12))

    if not boot:
        return np.nan, np.nan

    low, high = np.percentile(
        boot,
        [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100],
    )
    return float(low), float(high)


def sortino_ratio(r: pd.Series) -> float:
    """
    Annualized Sortino ratio: mean / downside std (only negative returns).

    Args:
        r: monthly returns.

    Returns:
        Annualized Sortino ratio, or NaN if undefined.
    """
    r = r.dropna()
    d = r[r < 0]
    if len(d) == 0:
        return np.nan
    dd = d.std()
    if dd <= 0:
        return np.nan
    return (r.mean() / dd) * np.sqrt(12)


def sortino_ci_annualized(
    r: pd.Series,
    n_boot: int = 2000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for annualized Sortino ratio.

    Args:
        r: monthly returns (Series)
        n_boot: number of bootstrap resamples
        ci: confidence level

    Returns:
        (low, high) bounds for annualized Sortino.
    """
    r = r.dropna().to_numpy()
    if len(r) < 3:
        return np.nan, np.nan

    rng = np.random.default_rng(7)
    boot = []

    for _ in range(n_boot):
        s = rng.choice(r, size=len(r), replace=True)
        d = s[s < 0]
        if len(d) == 0:
            continue
        dd = d.std()
        if dd > 0:
            boot.append((s.mean() / dd) * np.sqrt(12))

    if not boot:
        return np.nan, np.nan

    low, high = np.percentile(
        boot,
        [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100],
    )
    return float(low), float(high)


def series_metrics(r: pd.Series) -> dict:
    """
    Compute a set of performance metrics for a return series.

    Metrics:
      - mean_m: mean monthly return
      - ann_ret: annualized return
      - ann_vol: annualized volatility
      - sharpe: annualized Sharpe
      - sharpe_low/high: bootstrap CI for Sharpe
      - sortino: annualized Sortino
      - sortino_low/high: bootstrap CI for Sortino
      - max_dd: maximum drawdown
    """
    r = r.dropna()
    if len(r) == 0:
        # All NaNs → return a dictionary of NaNs
        keys = [
            "mean_m", "ann_ret", "ann_vol", "sharpe", "sharpe_low",
            "sharpe_high", "sortino", "sortino_low", "sortino_high", "max_dd",
        ]
        return {k: np.nan for k in keys}

    m = r.mean()
    v = r.std()

    ann_ret = (1 + m) ** 12 - 1
    ann_vol = v * np.sqrt(12)
    sharpe = (m / v) * np.sqrt(12) if v > 0 else np.nan
    s_lo, s_hi = sharpe_ci_annualized(r)
    sort = sortino_ratio(r)
    so_lo, so_hi = sortino_ci_annualized(r)

    return dict(
        mean_m=m,
        ann_ret=ann_ret,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sharpe_low=s_lo,
        sharpe_high=s_hi,
        sortino=sort,
        sortino_low=so_lo,
        sortino_high=so_hi,
        max_dd=max_drawdown(r),
    )


# ---------- Diebold–Mariano test helpers ----------

def hac_var(d: np.ndarray, lags: int) -> float:
    """
    HAC (Newey–West style) variance estimator for the mean of a series.

    Args:
        d: difference series (x_t - y_t)
        lags: number of lags to include in the kernel

    Returns:
        HAC estimate of var(mean(d)).
    """
    d = d - d.mean()
    T = len(d)
    gamma0 = np.dot(d, d) / T
    s = gamma0

    for k in range(1, min(lags, T - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * np.dot(d[:-k], d[k:]) / T

    return s / T


def diebold_mariano_test(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    """
    Diebold–Mariano test comparing two return series x and y via their difference.

    Args:
        x, y: two return Series indexed by month.

    Returns:
        (DM statistic, p-value) using a normal approximation.
        NaN values if the sample is too short.
    """
    x, y = x.dropna(), y.dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]

    d = (x - y).to_numpy()
    T = len(d)
    if T < 10:
        return np.nan, np.nan

    lags = int(T ** (1 / 3))
    var_mean = hac_var(d, lags)
    dm_stat = d.mean() / np.sqrt(var_mean)
    p_val = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_val)


# ---------- MAIN pipeline ----------

def main() -> None:
    # 1) Load LR and GB performance (net of costs)
    perf_lr = load_semicolon_csv(PERF_LR_PATH)
    perf_gb = load_semicolon_csv(PERF_GB_PATH)

    perf_lr["net_ret"] = pd.to_numeric(perf_lr["net_ret"], errors="coerce")
    perf_gb["net_ret"] = pd.to_numeric(perf_gb["net_ret"], errors="coerce")

    perf_lr = perf_lr[["month", "net_ret"]].rename(columns={"net_ret": "net_ret_lr"})
    perf_gb = perf_gb[["month", "net_ret"]].rename(columns={"net_ret": "net_ret_gb"})

    # 2) Load true ME deciles to build EW_10 benchmark
    true_ = load_semicolon_csv(TRUE_PANEL)
    true_.columns = [c.strip() for c in true_.columns]

    if "month" not in true_.columns:
        raise ValueError("true_panel_lr.csv must have a 'month' column")

    deciles = [c for c in true_.columns if c.startswith("ME")]
    for c in deciles:
        true_[c] = pd.to_numeric(true_[c], errors="coerce")

    true_["EW_10"] = true_[deciles].mean(axis=1)
    ew = true_[["month", "EW_10"]]

    # 3) Load FF file to build (total) Market return if available
    proc = load_semicolon_csv(PROC_FULL)
    proc.columns = [c.strip() for c in proc.columns]

    # Robust date handling: accept either 'date' or 'month'
    date_col = "date" if "date" in proc.columns else "month"
    proc["month"] = pd.to_datetime(proc[date_col], errors="coerce").dt.to_period("M").astype(str)

    market_available = ("Mkt-RF" in proc.columns) and ("RF" in proc.columns)
    if market_available:
        proc["Mkt-RF"] = pd.to_numeric(proc["Mkt-RF"], errors="coerce")
        proc["RF"] = pd.to_numeric(proc["RF"], errors="coerce")
        # Total market return = excess + risk-free
        proc["Market"] = proc["Mkt-RF"] + proc["RF"]
        mk = proc[["month", "Market"]]
    else:
        mk = pd.DataFrame(columns=["month", "Market"])

    # 4) Merge everything on month: LR, GB, EW_10, (optional) Market
    panel = (
        perf_lr
        .merge(perf_gb, on="month", how="inner")
        .merge(ew, on="month",   how="inner")
    )

    if market_available:
        panel = panel.merge(mk, on="month", how="left")

    panel = panel.sort_values("month").reset_index(drop=True)

    # 5) Compute metrics for each series
    series_cols = ["net_ret_lr", "net_ret_gb", "EW_10"]
    if "Market" in panel.columns:
        series_cols.append("Market")

    summary_rows = [
        dict(series=s, **series_metrics(panel[s]))
        for s in series_cols
    ]
    summary = pd.DataFrame(summary_rows)

    # 6) Diebold–Mariano tests: compare each strategy vs benchmarks
    tests = []

    def add_dm(name: str, a: pd.Series, b: pd.Series) -> None:
        dm, p = diebold_mariano_test(a, b)
        tests.append({"comparison": name, "DM_stat": dm, "p_value": p})

    # LR vs EW_10 / Market
    add_dm("LR_net vs EW_10", panel["net_ret_lr"], panel["EW_10"])
    if "Market" in panel.columns:
        add_dm("LR_net vs Market", panel["net_ret_lr"], panel["Market"])

    # GB vs EW_10 / Market
    add_dm("GB_net vs EW_10", panel["net_ret_gb"], panel["EW_10"])
    if "Market" in panel.columns:
        add_dm("GB_net vs Market", panel["net_ret_gb"], panel["Market"])

    # GB vs LR
    add_dm("GB_net vs LR_net", panel["net_ret_gb"], panel["net_ret_lr"])

    tests_df = pd.DataFrame(tests)

    # 7) Save all three CSV outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(panel, PANEL_OUT)
    write_aligned_csv(summary, SUMM_OUT)
    write_aligned_csv(tests_df, TESTS_OUT)

    print(f"Saved unified benchmarks panel   → {PANEL_OUT}")
    print(f"Saved unified benchmarks summary → {SUMM_OUT}")
    print(f"Saved unified DM tests           → {TESTS_OUT}")


if __name__ == "__main__":
    main()
