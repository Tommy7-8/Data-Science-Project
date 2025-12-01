# Predicting and Allocating Across Fama–French Size Portfolios with Machine Learning

## Student
Name: Tomas Papuga  
Student Number: 25111589

## Research Question
Do machine-learning forecasts of next-month return and volatility improve portfolio allocation across the ten Fama–French size portfolios (ME1–ME10) compared to Equal-Weight (EW10) and the Market?

## Setup

### Create environment
python -m venv .venv  
.venv\Scripts\activate  
pip install -r requirements.txt  

## Usage
python main.py  

## Runtime
Due to the large sample and the expanding-window walk-forward training for all ME1–ME10 portfolios (returns and volatility, LR and GB), a full run of the pipeline typically takes **around 1 hour** on a standard modern laptop.

## Expected Output
Full pipeline execution including:
- Data download from the Kenneth R. French Data Library  
- Feature construction (lagged returns, rolling volatility, factor lags)  
- Training LR and GB models for returns and volatility  
- Aggregating per-decile forecast metrics into `results/forecast_metrics/forecast_metrics_summary.txt`  
- Building prediction panels for ME1–ME10  
- Running LR and GB mean–variance allocations  
- Evaluating realized performance with turnover limits and transaction costs  
- Creating benchmark comparison tables and a cumulative-return plot  

All outputs are written into the `results/` directory.

## Project Structure
fama-french-ml-project/  
│  
├── main.py                          # Runs entire pipeline  
│  
├── data/  
│   ├── raw/                         # Raw Fama–French files  
│   └── processed/                   # Clean feature tables (returns, vol, factors)  
│  
├── src/  
│   ├── dataset/  
│   │   └── download_ff.py           # Download & clean FF data  
│   │  
│   ├── utils/  
│   │   ├── prepare_features_full.py # Build lag/vol/factor features  
│   │   ├── build_lr_panel.py        # LR prediction panel  
│   │   ├── build_gb_panels.py       # GB prediction panels  
│   │   └── summarise_forecast_metrics.py
│   │                                # Aggregate LR/GB forecast metrics (R², MAE, RMSE)
│   │  
│   ├── models/  
│   │   ├── train_all_lr.py          # LR return models  
│   │   ├── train_all_vol_lr.py      # LR volatility models  
│   │   ├── train_all_gb.py          # GB return models  
│   │   └── train_all_gb_vol.py      # GB volatility models  
│   │  
│   ├── alloc/  
│   │   ├── run_allocation_lr.py     # LR allocation  
│   │   ├── run_allocation_gb.py     # GB allocation  
│   │   ├── evaluate_allocation_lr.py  
│   │   ├── evaluate_allocation_gb.py  
│   │   └── compare_allocation_summaries.py  
│   │  
│   └── benchmark/  
│       ├── compare_benchmarks.py    # Summary tables + DM tests  
│       └── plot_cumulative_returns.py  
│  
└── results/  
    ├── oos_panel_lr/                # LR prediction panels (returns / vol)  
    ├── oos_panel_gb/                # GB prediction panels (returns / vol)  
    ├── alloc_lr/                    # LR allocation weights and returns  
    ├── alloc_gb/                    # GB allocation weights and returns  
    ├── alloc_comparison/            # LR vs GB allocation summaries  
    ├── forecast_metrics/            # LR vs GB forecast metrics summary  
    └── benchmark/                   # Benchmark panels, summary stats, plots  

## Results
- LR and GB out-of-sample forecasts (returns and volatility)  
- Aggregated LR vs GB forecast metrics in `results/forecast_metrics/forecast_metrics_summary.txt`  
- Mean–variance allocations for LR and GB  
- Net portfolio returns after turnover limits and transaction costs  
- Benchmark comparison versus EW10 and the Market  
- Final cumulative-return plot stored at: `results/benchmark/cumulative_returns.png`
