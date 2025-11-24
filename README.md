# Predicting and Allocating Across Fama–French Size Portfolios with Machine Learning
# Name: Papuga Tomas

This project studies return and risk patterns of U.S. equities sorted by firm size using monthly data from the Kenneth R. French Data Library.

The pipeline:

Downloads & cleans Fama–French size portfolios and FF3 factors

Builds feature tables with lagged returns, volatility measures, and factor lags

Trains two model families:

Linear Ridge Regression (LR)

Gradient Boosting (GB)
for predicting next-month return and volatility

Performs walk-forward out-of-sample predictions for ME1…ME10

Builds prediction panels for LR and GB

Runs mean–variance allocation (long-only, weight caps, shrinkage covariance)

Evaluates portfolio performance including turnover and transaction costs

Compares LR vs GB

Benchmarks against EW10 and Market

Produces final benchmark tables and cumulative return plot

Everything is fully automated and reproducible.

1. Data

Source: Kenneth R. French Data Library

Portfolios Formed on Size (ME1–ME10)

Fama–French 3 Factors (Mkt-RF, SMB, HML, RF)

Main processed file:

data/processed/ff_size_deciles_with_ff3_monthly_decimals.csv

2. Modeling Framework

Return and volatility are predicted one month ahead.

Predictive targets (for every decile ME1…ME10):

Next-month return

Next-month volatility (LR uses squared returns; GB uses volatility features)

Features:

Lagged returns: MEj_lag1 … MEj_lag12

Rolling realized volatility: MEj_vol_3m, MEj_vol_6m, MEj_vol_12m

Factor lags: Mkt-RF_lag1, SMB_lag1

RF kept separately

Models used:

Linear Ridge Regression (LR):

StandardScaler + Ridge

Walk-forward expanding window

CV inside training window

Alpha chosen monthly

Gradient Boosting (GB):

HistGradientBoostingRegressor

Static mode (fit once) or walkforward mode

Non-linear, faster

Walk-forward process:

Use first 240 months to train

Predict next month

Expand window and repeat
Prevents look-ahead bias and simulates real-time trading.

3. Portfolio Allocation

Allocator = mean–variance optimizer with:

Long-only

Fully invested

Max weight per asset = 0.40

Covariance window = 120 months

Shrinkage toward diagonal

Predicted variance blended into covariance diagonal

Monthly rebalancing

Turnover limit = 20%

Transaction cost = 0.10% per 100% turnover

Two portfolios:

LR portfolio (uses LR predictions)

GB portfolio (uses GB predictions)

4. Benchmark Comparison

Benchmarks:

EW_10 (equal-weight across ME1…ME10)

Market = Mkt-RF + RF (if available)

Metrics:

Monthly mean return

Annualized return

Annualized volatility

Sharpe ratio with 95% CI (bootstrap)

Sortino ratio with 95% CI

Maximum drawdown

Diebold–Mariano tests:

LR vs EW_10

LR vs Market

GB vs EW_10

GB vs Market

GB vs LR

Output folder:

results/benchmark
benchmarks_panel.csv
benchmarks_summary.csv
benchmarks_tests.csv
cumulative_returns.png

5. Project Structure

data/
raw/ (raw FF data)
processed/ (clean, aligned, feature files)

results/
oos_panel_lr/
oos_vol_panel_lr/
oos_preds_lr/
oos_vol_preds_lr/

oos_preds_gb/
oos_vol_gb/
gb_panels_returns/
gb_panels_vol/

alloc_lr/ (LR allocation and performance)
alloc_gb/ (GB allocation and performance)

alloc_comparison/
benchmark/

src/
dataset/
download_ff.py
utils/
prepare_features_full.py
build_lr_panel.py
build_gb_panels.py
models/
train_all_lr.py
train_all_vol.py
train_all_gb.py
train_all_gb_vol.py
alloc/
run_allocation_lr.py
run_allocation_gb.py
evaluate_allocation.py
evaluate_allocation_gb.py
compare_allocation_summaries.py
benchmark/
compare_benchmarks.py
plot_cumulative_returns.py

6. Installation

pip install numpy pandas scikit-learn matplotlib scipy

7. Pipeline — Full Step-by-Step Instructions

Run these from the project root.

Download Fama–French data
python src/dataset/download_ff.py

Prepare feature tables
python src/utils/prepare_features_full.py

Train LR models (return + vol)
python src/models/train_all_lr.py
python src/models/train_all_vol.py

Build LR prediction panels
python src/utils/build_lr_panel.py

Train GB models (return + vol)
python src/models/train_all_gb.py
python src/models/train_all_gb_vol.py

Build GB prediction panels
python src/utils/build_gb_panels.py

Run LR and GB allocations
python src/alloc/run_allocation_lr.py
python src/alloc/run_allocation_gb.py

Evaluate portfolio performance
python src/alloc/evaluate_allocation.py
python src/alloc/evaluate_allocation_gb.py

Compare LR vs GB summaries
python src/alloc/compare_allocation_summaries.py

Benchmark comparison
python src/benchmark/compare_benchmarks.py

Plot cumulative returns
python src/benchmark/plot_cumulative_returns.py

8. Research Question

Do machine-learning forecasts of return and volatility on size-sorted portfolios improve portfolio outcomes compared to simple benchmarks like equal-weight or the market?

This project answers it using:

Rigorous walk-forward forecasting

Two fundamentally different ML models

Realistic costs & turnover limits

Mean–variance allocation with shrinkage

Statistical tests and confidence intervals

9. Notes

All CSVs inside data/ are ignored by git via .gitignore

All results are reproducible by running the pipeline

No model pickle files are saved (design choice)

All CSV outputs are visually aligned with semicolon separators

10. Possible Future Extensions

Rolling performance charts

Add LGBM, XGBoost

Add more FF portfolios (value, momentum, profitability)

Try Hierarchical Risk Parity

Try online learning for faster walk-forward updates