# src/benchmark/plot_cumulative_returns.py
# Plot cumulative returns for:
#   - Baseline LR portfolio (net)
#   - GB portfolio (net)
#   - EW_10
#   - Market (if available)
#
# Input:  results/benchmark/benchmarks_panel.csv
# Output: results/benchmark/cumulative_returns.png

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick  # <-- NEW

PANEL_PATH = Path("results/benchmark/benchmarks_panel.csv")
OUT_FIG    = Path("results/benchmark/cumulative_returns.png")


def main():
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing benchmarks panel: {PANEL_PATH}")

    df = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Convert numeric columns
    for c in df.columns:
        if c != "month":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["month"]).sort_values("month")

    # Convert YYYY-MM to datetime (day=1)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["month"])

    if df.empty:
        raise ValueError("No valid rows in benchmarks_panel.csv after parsing 'month'.")

    # ----- Restrict to 1980 onwards -----
    cutoff = pd.Timestamp(1980, 1, 1)
    df = df[df["month"] >= cutoff].copy()
    if df.empty:
        raise ValueError("No data available from 1980 onwards in benchmarks_panel.csv.")
    # ------------------------------------

    # Compute cumulative returns (growth of 1)
    ret_cols = [c for c in df.columns if c != "month"]
    for c in ret_cols:
        df[f"cum_{c}"] = (1.0 + df[c]).cumprod()

    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    # --- Plot main lines (only if they exist) ---
    if "cum_net_ret_lr" in df.columns:
        ax.plot(df["month"], df["cum_net_ret_lr"],
                label="ML Portfolio (LR, Net)", linewidth=2.8)

    if "cum_net_ret_gb" in df.columns:
        ax.plot(df["month"], df["cum_net_ret_gb"],
                label="GB Portfolio (Net)", linewidth=2.5)

    if "cum_EW_10" in df.columns:
        ax.plot(df["month"], df["cum_EW_10"],
                label="Equal-Weight (EW_10)", linewidth=2.0)

    if "cum_Market" in df.columns:
        ax.plot(df["month"], df["cum_Market"],
                label="Market (Mkt-RF + RF)", linewidth=2.0, linestyle="--")

    # --- Formatting ---
    ax.set_title("Cumulative Returns: LR vs GB vs Benchmarks (from 1980)",
                 fontsize=14, weight="bold", pad=10)
    ax.set_ylabel("Growth of 1")
    ax.set_xlabel("Year")

    # More precise Y axis: more ticks + 2 decimals
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=20))          # more tick marks
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))   # 2 decimal places

    ax.grid(alpha=0.25)

    # X-axis: show only years (every 5 years)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_xlim(df["month"].min(), df["month"].max())
    plt.xticks(rotation=0)

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"Saved cumulative returns chart → {OUT_FIG}")


if __name__ == "__main__":
    main()
