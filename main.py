"""
main.py

Single entry point to run the whole project pipeline.

It sequentially executes the existing scripts under src/:

1) Download & prepare data
2) Train LR & GB return models + vol models
3) Build prediction panels
4) Run allocations and evaluations
5) Run benchmark comparisons and plots
"""

import logging
import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_step(description: str, module: str):
    """
    Run a module as if it were called with `python -m module`.
    Example: module='src.dataset.download_ff'
    """
    logging.info(f"=== {description} ({module}) ===")
    runpy.run_module(module, run_name="__main__")


def run_full_pipeline():
    # 1) Data
    run_step("Downloading / updating Fama-French data",
             "src.dataset.download_ff")

    # 2) Feature engineering
    run_step("Preparing full feature set",
             "src.utils.prepare_features_full")

    # 3) Train return models (LR & GB)
    run_step("Training LR return models for ME1–ME10",
             "src.models.train_all_lr")
    run_step("Training GB return models for ME1–ME10",
             "src.models.train_all_gb")

    # 4) Train volatility models (LR & GB)
    run_step("Training LR volatility models",
             "src.models.train_all_vol_lr")
    run_step("Training GB volatility models",
             "src.models.train_all_gb_vol")

    # 5) Build prediction panels
    run_step("Building LR prediction panels",
             "src.utils.build_lr_panel")
    run_step("Building GB prediction panels",
             "src.utils.build_gb_panels")

    # 6) Run allocations
    run_step("Running allocation using LR forecasts",
             "src.alloc.run_allocation_lr")
    run_step("Running allocation using GB forecasts",
             "src.alloc.run_allocation_gb")

    # 7) Evaluate allocations
    run_step("Evaluating LR allocation",
             "src.alloc.evaluate_allocation_lr")
    run_step("Evaluating GB allocation",
             "src.alloc.evaluate_allocation_gb")
    run_step("Comparing LR vs GB allocation summaries",
             "src.alloc.compare_allocation_summaries")

    # 8) Benchmarks / plots
    run_step("Comparing strategy vs benchmarks",
             "src.benchmark.compare_benchmarks")
    run_step("Plotting cumulative returns",
             "src.benchmark.plot_cumulative_returns")

    logging.info("✅ Full pipeline finished successfully.")


def main():
    setup_logging()
    logging.info(f"Project root: {PROJECT_ROOT}")
    run_full_pipeline()


if __name__ == "__main__":
    main()
