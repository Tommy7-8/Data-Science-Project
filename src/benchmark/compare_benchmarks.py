# src/benchmark/compare_benchmarks.py
# Compare Baseline (ML allocation, net of costs) vs. multiple benchmarks:
#  - EW_10 (equal-weight)
#  - SMB (official Fama–French)
#  - Market (optional)
# Outputs aligned semicolon CSVs + JSON summary and DM tests.
# Uses annualized Sharpe with 95% CI and HAC/Newey–West DM tests.

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

PERF_PATH   = Path("results/alloc/performance_baseline.csv")
TRUE_PANEL  = Path("results/oos_panel/true_panel.csv")
PROC_FULL   = Path("data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv")

OUT_DIR     = Path("results/benchmark")
PANEL_OUT   = OUT_DIR / "benchmarks_panel.csv"
SUMM_OUT    = OUT_DIR / "benchmarks_summary.csv"
TESTS_OUT   = OUT_DIR / "benchmarks_tests.csv"
META_OUT    = OUT_DIR / "benchmarks_meta.json"

def _fmt_num(val, decimals=6):
    if pd.isna(val): 
        return ""
    return f"{float(val):.{decimals}f}"

def write_aligned_csv(df: pd.DataFrame, path: Path, decimals=6):
    df_str = df.copy()
    for c in df_str.columns:
        if c != "month":
            ser = pd.to_numeric(df_str[c], errors="coerce")
            df_str[c] = ser.map(lambda v: _fmt_num(v, decimals)) if ser.notna().any() else df_str[c]
    widths = {c: max(len(c), df_str[c].astype(str).map(len).max()) for c in df_str.columns}
    header = ";".join([f"{c:<{widths[c]}}" if c == "month" else f"{c:>{widths[c]}}" for c in df_str.columns])
    lines = [header]
    for _, r in df_str.iterrows():
        line = ";".join([f"{r[c]:<{widths[c]}}" if c == "month" else f"{r[c]:>{widths[c]}}" for c in df_str.columns])
        lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8-sig")

def max_drawdown(r):
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())

def sharpe_ci_annualized(r, n_boot=2000, ci=0.95):
    r = r.dropna().to_numpy()
    if len(r) < 3: 
        return np.nan, np.nan
    rng = np.random.default_rng(7)
    boot = []
    for _ in range(n_boot):
        s = rng.choice(r, size=len(r), replace=True)
        if s.std() > 0:
            boot.append((s.mean() / s.std()) * np.sqrt(12))
    low, high = np.percentile(boot, [(1-ci)/2*100, (1+ci)/2*100])
    return float(low), float(high)

def series_metrics(r):
    r = r.dropna()
    if len(r) == 0:
        return dict(mean_m=np.nan, ann_ret=np.nan, ann_vol=np.nan, sharpe=np.nan,
                    sharpe_low=np.nan, sharpe_high=np.nan, max_dd=np.nan)
    m, v = r.mean(), r.std()
    ann_ret = (1 + m)**12 - 1
    ann_vol = v * np.sqrt(12)
    sharpe = (m / v) * np.sqrt(12) if v > 0 else np.nan
    low, high = sharpe_ci_annualized(r)
    return dict(mean_m=m, ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
                sharpe_low=low, sharpe_high=high, max_dd=max_drawdown(r))

def hac_var(d, lags):
    d = d - d.mean()
    T = len(d)
    gamma0 = np.dot(d, d) / T
    s = gamma0
    for k in range(1, min(lags, T-1)+1):
        w = 1 - k/(lags+1)
        s += 2*w*np.dot(d[:-k], d[k:]) / T
    return s / T

def diebold_mariano_test(x, y):
    x, y = x.dropna(), y.dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    d = (x - y).to_numpy()
    T = len(d)
    if T < 10: 
        return np.nan, np.nan
    var_mean = hac_var(d, int(T**(1/3)))
    dm = d.mean() / np.sqrt(var_mean)
    p = 2*(1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)

def load_csv(path):
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig", engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

def main():
    perf = load_csv(PERF_PATH)
    true_ = load_csv(TRUE_PANEL)
    proc = load_csv(PROC_FULL)

    perf["net_ret"] = pd.to_numeric(perf["net_ret"], errors="coerce")

    deciles = [c for c in true_.columns if c.startswith("ME")]
    for c in deciles:
        true_[c] = pd.to_numeric(true_[c], errors="coerce")

    # Equal-weight of all 10 ME deciles
    true_["EW_10"] = true_[deciles].mean(axis=1)
    true_keep = ["month", "EW_10"]

    date_col = "date" if "date" in proc.columns else "month"
    proc["month"] = pd.to_datetime(proc[date_col]).dt.to_period("M").astype(str)
    proc["SMB"] = pd.to_numeric(proc["SMB"], errors="coerce")
    if "Mkt-RF" in proc.columns and "RF" in proc.columns:
        proc["Mkt-RF"] = pd.to_numeric(proc["Mkt-RF"], errors="coerce")
        proc["RF"]     = pd.to_numeric(proc["RF"], errors="coerce")
        proc["Market"] = proc["Mkt-RF"] + proc["RF"]

    # Merge all series
    panel = perf[["month", "net_ret"]].merge(true_[true_keep], on="month", how="inner").merge(
        proc[["month", "SMB"]], on="month", how="inner"
    )
    if "Market" in proc.columns:
        panel = panel.merge(proc[["month", "Market"]], on="month", how="left")

    # Metrics
    series_cols = ["net_ret", "EW_10", "SMB"] + (["Market"] if "Market" in panel.columns else [])
    rows = [dict(series=s, **series_metrics(panel[s])) for s in series_cols]
    summary = pd.DataFrame(rows)

    # DM tests vs Baseline
    base = panel["net_ret"]
    comps = []
    for c in [x for x in ["EW_10", "SMB", "Market"] if x in panel.columns]:
        DM, p = diebold_mariano_test(base, panel[c])
        comps.append(dict(comparison=f"Baseline_net vs {c}", DM_stat=DM, p_value=p))
    tests = pd.DataFrame(comps)

    # Save
    write_aligned_csv(panel, PANEL_OUT)
    write_aligned_csv(summary, SUMM_OUT)
    write_aligned_csv(tests, TESTS_OUT)
    json.dump(
        dict(
            notes="Benchmarks: EW_10, official SMB, optional Market; annualized Sharpe 95% CI; DM HAC/Newey-West."
        ),
        open(META_OUT, "w", encoding="utf-8"),
        indent=2
    )

    print(f"Saved results → {OUT_DIR}")
    print("\nAnnualized Sharpe (95% CI):")
    for _, r in summary.iterrows():
        print(f" {r['series']:>10s} | Sharpe {r['sharpe']:.3f} (CI [{r['sharpe_low']:.3f},{r['sharpe_high']:.3f}])")

if __name__ == "__main__":
    main()
