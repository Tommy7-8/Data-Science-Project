# src/benchmark/compare_benchmarks.py
# Unified benchmark comparison for allocation strategies:
#  - Baseline LR portfolio (net of costs)
#  - GB portfolio (net of costs)
#  - EW_10 (equal-weight of 10 size deciles)
#  - Market (if available)
#
# Outputs (all in results/benchmark/):
#   - benchmarks_panel.csv   : monthly returns
#   - benchmarks_summary.csv : Sharpe/Sortino with CIs, max DD
#   - benchmarks_tests.csv   : Diebold–Mariano tests

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ---- Paths ----
PERF_LR_PATH = Path("results/alloc/performance_baseline.csv")
PERF_GB_PATH = Path("results/alloc_gb/performance_gb_mv.csv")
TRUE_PANEL   = Path("results/oos_panel/true_panel.csv")
PROC_FULL    = Path("data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv")

OUT_DIR      = Path("results/benchmark")
PANEL_OUT    = OUT_DIR / "benchmarks_panel.csv"
SUMM_OUT     = OUT_DIR / "benchmarks_summary.csv"
TESTS_OUT    = OUT_DIR / "benchmarks_tests.csv"


# ---------- Helpers ----------

def _fmt_num(val, decimals=6):
    if pd.isna(val):
        return ""
    return f"{float(val):.{decimals}f}"


def write_aligned_csv(df: pd.DataFrame, path: Path, decimals=6, sep=";"):
    df_str = df.copy()
    for c in df_str.columns:
        if c != "month":
            ser = pd.to_numeric(df_str[c], errors="coerce")
            if ser.notna().any():
                df_str[c] = ser.map(lambda v: _fmt_num(v, decimals))
    widths = {c: max(len(c), df_str[c].astype(str).map(len).max()) for c in df_str.columns}

    header = sep.join(
        f"{c:<{widths[c]}}" if c == "month" else f"{c:>{widths[c]}}"
        for c in df_str.columns
    )
    lines = [header]

    for _, r in df_str.iterrows():
        line = sep.join(
            f"{r[c]:<{widths[c]}}" if c == "month" else f"{r[c]:>{widths[c]}}"
            for c in df_str.columns
        )
        lines.append(line)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")


def load_semicolon_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df


def max_drawdown(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def sharpe_ci_annualized(r: pd.Series, n_boot=2000, ci=0.95):
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
    if len(boot) == 0:
        return np.nan, np.nan
    low, high = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(low), float(high)


def sortino_ratio(r: pd.Series):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    downside = r[r < 0]
    if len(downside) == 0:
        return np.nan
    dd = downside.std()
    if dd <= 0:
        return np.nan
    return (r.mean() / dd) * np.sqrt(12)


def sortino_ci_annualized(r: pd.Series, n_boot=2000, ci=0.95):
    r = r.dropna().to_numpy()
    if len(r) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(n_boot):
        s = rng.choice(r, size=len(r), replace=True)
        downside = s[s < 0]
        if len(downside) == 0:
            continue
        dd = downside.std()
        if dd > 0:
            boot.append((s.mean() / dd) * np.sqrt(12))
    if len(boot) == 0:
        return np.nan, np.nan
    low, high = np.percentile(boot, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(low), float(high)


def series_metrics(r: pd.Series):
    r = r.dropna()
    if len(r) == 0:
        return dict(
            mean_m=np.nan,
            ann_ret=np.nan,
            ann_vol=np.nan,
            sharpe=np.nan,
            sharpe_low=np.nan,
            sharpe_high=np.nan,
            sortino=np.nan,
            sortino_low=np.nan,
            sortino_high=np.nan,
            max_dd=np.nan,
        )
    m, v = r.mean(), r.std()
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


def hac_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    T = len(d)
    gamma0 = np.dot(d, d) / T
    s = gamma0
    for k in range(1, min(lags, T - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * np.dot(d[:-k], d[k:]) / T
    return s / T


def diebold_mariano_test(x: pd.Series, y: pd.Series):
    # DM test on difference of returns x - y
    x, y = x.dropna(), y.dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).to_numpy()
    T = len(d)
    if T < 10:
        return np.nan, np.nan
    var_mean = hac_var(d, int(T ** (1 / 3)))
    dm = d.mean() / np.sqrt(var_mean)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


# ---------- Main ----------

def main():
    # 1) Load LR and GB performance (net of costs)
    perf_lr = load_semicolon_csv(PERF_LR_PATH)
    perf_gb = load_semicolon_csv(PERF_GB_PATH)

    perf_lr["net_ret"] = pd.to_numeric(perf_lr["net_ret"], errors="coerce")
    perf_gb["net_ret"] = pd.to_numeric(perf_gb["net_ret"], errors="coerce")

    perf_lr = perf_lr[["month", "net_ret"]].rename(columns={"net_ret": "net_ret_lr"})
    perf_gb = perf_gb[["month", "net_ret"]].rename(columns={"net_ret": "net_ret_gb"})

    # 2) Load true deciles to build EW_10
    true_ = load_semicolon_csv(TRUE_PANEL)
    true_.columns = [c.strip() for c in true_.columns]
    if "month" not in true_.columns:
        raise ValueError("true_panel.csv must have a 'month' column")

    deciles = [c for c in true_.columns if c.startswith("ME")]
    for c in deciles:
        true_[c] = pd.to_numeric(true_[c], errors="coerce")
    true_["EW_10"] = true_[deciles].mean(axis=1)
    ew = true_[["month", "EW_10"]]

    # 3) Load FF file to build Market
    proc = load_semicolon_csv(PROC_FULL)
    proc.columns = [c.strip() for c in proc.columns]
    date_col = "date" if "date" in proc.columns else "month"
    proc["month"] = pd.to_datetime(proc[date_col]).dt.to_period("M").astype(str)

    market_available = ("Mkt-RF" in proc.columns) and ("RF" in proc.columns)
    if market_available:
        proc["Mkt-RF"] = pd.to_numeric(proc["Mkt-RF"], errors="coerce")
        proc["RF"]     = pd.to_numeric(proc["RF"], errors="coerce")
        proc["Market"] = proc["Mkt-RF"] + proc["RF"]
        mk = proc[["month", "Market"]]
    else:
        mk = pd.DataFrame(columns=["month", "Market"])

    # 4) Merge everything on month
    panel = (
        perf_lr
        .merge(perf_gb, on="month", how="inner")
        .merge(ew, on="month", how="inner")
    )

    if market_available:
        panel = panel.merge(mk, on="month", how="left")

    panel = panel.sort_values("month").reset_index(drop=True)

    # 5) Compute metrics for each series
    series_cols = ["net_ret_lr", "net_ret_gb", "EW_10"]
    if "Market" in panel.columns:
        series_cols.append("Market")

    rows = [dict(series=s, **series_metrics(panel[s])) for s in series_cols]
    summary = pd.DataFrame(rows)

    # 6) DM tests:
    #    LR vs EW_10/Market, GB vs EW_10/Market, GB vs LR
    tests = []
    # helper closure
    def add_dm(name, a, b):
        dm, p = diebold_mariano_test(a, b)
        tests.append(dict(comparison=name, DM_stat=dm, p_value=p))

    # LR vs benchmarks
    add_dm("LR_net vs EW_10", panel["net_ret_lr"], panel["EW_10"])
    if "Market" in panel.columns:
        add_dm("LR_net vs Market", panel["net_ret_lr"], panel["Market"])

    # GB vs benchmarks
    add_dm("GB_net vs EW_10", panel["net_ret_gb"], panel["EW_10"])
    if "Market" in panel.columns:
        add_dm("GB_net vs Market", panel["net_ret_gb"], panel["Market"])

    # GB vs LR
    add_dm("GB_net vs LR_net", panel["net_ret_gb"], panel["net_ret_lr"])

    tests_df = pd.DataFrame(tests)

    # 7) Save three CSVs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_aligned_csv(panel, PANEL_OUT)
    write_aligned_csv(summary, SUMM_OUT)
    write_aligned_csv(tests_df, TESTS_OUT)

    print(f"Saved unified benchmarks panel   → {PANEL_OUT}")
    print(f"Saved unified benchmarks summary → {SUMM_OUT}")
    print(f"Saved unified DM tests           → {TESTS_OUT}")

    print("\nAnnualized Sharpe (95% CI) and Sortino (95% CI):")
    for _, r in summary.iterrows():
        print(
            f" {r['series']:>10s} | Sharpe {r['sharpe']:.3f} "
            f"(CI [{r['sharpe_low']:.3f},{r['sharpe_high']:.3f}])"
            f" | Sortino {r['sortino']:.3f} "
            f"(CI [{r['sortino_low']:.3f},{r['sortino_high']:.3f}])"
        )


if __name__ == "__main__":
    main()
