# src/benchmark/plot_cumulative_returns.py
#
# Plot cumulative returns for:
#   - Baseline LR portfolio (net returns)
#   - GB portfolio (net returns)
#   - EW_10 benchmark
#   - Market benchmark (if FF data provides Mkt-RF + RF)
#
# Input:
#   results/benchmark/benchmarks_panel.csv
#
# Output:
#   results/benchmark/cumulative_returns.png
#
# The goal is simply to visualise the investment growth of €1 (or $1)
# for all strategies since 1980, using the cleaned benchmark panel.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick


# Location of the merged benchmark panel created earlier
PANEL_PATH = Path("results/benchmark/benchmarks_panel.csv")

# Output figure path
OUT_FIG = Path("results/benchmark/cumulative_returns.png")


def main() -> None:
    """Load the benchmark panel, compute cumulative returns,
    and save a clean cumulative return plot."""

    # ---- Load file and basic validation ----
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing benchmarks panel: {PANEL_PATH}")

    df = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig")
    # Clean header cells from whitespace / BOM
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # Convert all data columns (except month) to numeric
    for c in df.columns:
        if c != "month":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["month"]).sort_values("month")

    # Convert YYYY-MM string into datetime (use day=1 convention)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m", errors="coerce")
    df = df.dropna(subset=["month"])

    if df.empty:
        raise ValueError("benchmarks_panel.csv contains no valid 'month' rows.")

    # ---- Restrict analysis to post-1980 period ----
    cutoff = pd.Timestamp(1980, 1, 1)
    df = df[df["month"] >= cutoff].copy()
    if df.empty:
        raise ValueError("No rows available from 1980 onward.")

    # ---- Compute cumulative returns: growth of a 1€ investment ----
    ret_cols = [c for c in df.columns if c != "month"]
    for c in ret_cols:
        df[f"cum_{c}"] = (1 + df[c]).cumprod()

    # ---- Create figure ----
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    # Only plot lines that exist in the panel
    if "cum_net_ret_lr" in df.columns:
        ax.plot(
            df["month"], df["cum_net_ret_lr"],
            label="ML Portfolio (LR, Net)",
            linewidth=2.8
        )

    if "cum_net_ret_gb" in df.columns:
        ax.plot(
            df["month"], df["cum_net_ret_gb"],
            label="GB Portfolio (Net)",
            linewidth=2.5
        )

    if "cum_EW_10" in df.columns:
        ax.plot(
            df["month"], df["cum_EW_10"],
            label="Equal-Weight (EW_10)",
            linewidth=2.0
        )

    if "cum_Market" in df.columns:
        ax.plot(
            df["month"], df["cum_Market"],
            label="Market (Mkt-RF + RF)",
            linewidth=2.0,
            linestyle="--"
        )

    # ---- Figure formatting ----
    ax.set_title(
        "Cumulative Returns: LR vs GB vs Benchmarks (from 1980)",
        fontsize=14,
        weight="bold",
        pad=10
    )
    ax.set_ylabel("Growth of 1")
    ax.set_xlabel("Year")

    # More informative y-axis: many ticks + two decimals
    ax.yaxis.set_major_locator(mtick.MaxNLocator(nbins=20))
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    ax.grid(alpha=0.25)

    # X-axis: tick every 5 years
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_xlim(df["month"].min(), df["month"].max())
    plt.xticks(rotation=0)

    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    # ---- Save plot ----
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()

    print(f"Saved cumulative returns chart → {OUT_FIG}")


if __name__ == "__main__":
    main()
