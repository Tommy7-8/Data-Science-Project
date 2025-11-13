# src/benchmark/plot_cumulative_returns.py
# Plot cumulative returns for ML Portfolio (net) vs EW_10, SMB, Market.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates

PANEL_PATH = Path("results/benchmark/benchmarks_panel.csv")
OUT_FIG    = Path("results/benchmark/cumulative_returns.png")

def main():
    df = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for c in df.columns:
        if c != "month":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["month"]).sort_values("month")

    # Convert YYYY-MM string to a proper datetime (day set to 1, but we only show years)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")

    # Compute cumulative returns (growth of $1)
    cols = [c for c in df.columns if c != "month"]
    for c in cols:
        df[f"cum_{c}"] = (1 + df[c]).cumprod()

    plt.figure(figsize=(11, 6))
    plt.style.use("seaborn-v0_8-muted")

    ax = plt.gca()

    # --- Plot main lines (only series that exist) ---
    if "cum_net_ret" in df.columns:
        ax.plot(df["month"], df["cum_net_ret"], label="ML Portfolio (Net)", linewidth=2.8)
    if "cum_EW_10" in df.columns:
        ax.plot(df["month"], df["cum_EW_10"], label="Equal-Weight (EW_10)", linewidth=2.2)
    if "cum_SMB" in df.columns:
        ax.plot(df["month"], df["cum_SMB"], label="SMB (Fama–French)", linewidth=1.8, linestyle="--")
    if "cum_Market" in df.columns:
        ax.plot(df["month"], df["cum_Market"], label="Market (Mkt-RF+RF)", linewidth=2.2)

    # --- Formatting ---
    ax.set_title("Cumulative Returns: ML Portfolio vs Benchmarks",
                 fontsize=14, weight="bold", pad=10)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Year")
    ax.grid(alpha=0.25)

    # X-axis: show only years as labels
    # YearLocator(1) -> every year; change to YearLocator(5) if you want every 5 years.
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_xlim(df["month"].min(), df["month"].max())
    plt.xticks(rotation=0)

    # Legend and layout
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()

    print(f"Saved cumulative returns chart → {OUT_FIG}")

if __name__ == "__main__":
    main()
