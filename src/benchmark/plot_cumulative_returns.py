# src/benchmark/plot_cumulative_returns.py
# Plot cumulative returns for ML Portfolio (net) vs EW_10, ME1, ME10, SMB, Market.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PANEL_PATH = Path("results/benchmark/benchmarks_panel.csv")
OUT_FIG    = Path("results/benchmark/cumulative_returns.png")

def main():
    df = pd.read_csv(PANEL_PATH, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    for c in df.columns:
        if c != "month":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["month"]).sort_values("month")
    # Convert to datetime for plotting
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")

    # Compute cumulative returns (growth of $1)
    cols = [c for c in df.columns if c != "month"]
    for c in cols:
        df[f"cum_{c}"] = (1 + df[c]).cumprod()

    plt.figure(figsize=(11, 6))
    plt.style.use("seaborn-v0_8-muted")

    # --- Plot main lines ---
    plt.plot(df["month"], df["cum_net_ret"], label="ML Portfolio (Net)", linewidth=2.8, color="#2E86C1")
    plt.plot(df["month"], df["cum_EW_10"], label="Equal-Weight (EW_10)", linewidth=2.2, color="#17A589")
    plt.plot(df["month"], df["cum_ME1"], label="Smallest Decile (ME1)", linewidth=2.0, color="#CA6F1E")
    plt.plot(df["month"], df["cum_ME10"], label="Largest Decile (ME10)", linewidth=2.0, color="#884EA0")
    if "cum_SMB" in df.columns:
        plt.plot(df["month"], df["cum_SMB"], label="SMB (Fama–French)", linewidth=1.8, linestyle="--", color="#566573")
    if "cum_Market" in df.columns:
        plt.plot(df["month"], df["cum_Market"], label="Market (Mkt-RF+RF)", linewidth=2.2, color="#B03A2E")

    # --- Formatting ---
    plt.title("Cumulative Returns: ML Portfolio vs Benchmarks", fontsize=14, weight="bold", pad=10)
    plt.ylabel("Growth of $1")
    plt.xlabel("Year")
    plt.grid(alpha=0.25)

    # X-axis ticks every 5 years
    years = pd.to_datetime(df["month"]).dt.year
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.xticks(rotation=0)
    plt.xlim(df["month"].min(), df["month"].max())

    # Legend and layout
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300)
    plt.show()

    print(f"Saved better chart → {OUT_FIG}")

if __name__ == "__main__":
    main()
