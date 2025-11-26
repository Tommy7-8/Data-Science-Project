from pathlib import Path

# Resolve project root as two levels above this file: src/utils -> src -> project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"

DIR_LR_RET = RESULTS_ROOT / "reports_lr"
DIR_GB_RET = RESULTS_ROOT / "reports_gb"
DIR_LR_VOL = RESULTS_ROOT / "reports_vol_lr"
DIR_GB_VOL = RESULTS_ROOT / "reports_gb_vol"

# Subfolder for the summary output
OUTPUT_DIR = RESULTS_ROOT / "forecast_metrics"


def parse_report(path: Path):
    """
    Parse a single text report and extract oos_r2, oos_mae, oos_rmse.
    Returns a dict with keys 'R2', 'MAE', 'RMSE'. Missing values are None.
    """
    metrics = {"R2": None, "MAE": None, "RMSE": None}
    if not path.exists():
        return metrics

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().lower()
            if line.startswith("oos_r2"):
                try:
                    metrics["R2"] = float(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("oos_mae"):
                try:
                    metrics["MAE"] = float(line.split(":", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("oos_rmse"):
                try:
                    metrics["RMSE"] = float(line.split(":", 1)[1])
                except ValueError:
                    pass
    return metrics


def gather_rows(target: str, dir_lr: Path, dir_gb: Path, lr_tpl: str, gb_tpl: str):
    """
    Build a list of rows for a given target ('return' or 'vol').
    Each row: {'target','decile','model','R2','MAE','RMSE'}.
    """
    rows = []
    for i in range(1, 11):
        decile = f"ME{i}"
        lr_path = dir_lr / lr_tpl.format(decile=decile)
        gb_path = dir_gb / gb_tpl.format(decile=decile)

        lr = parse_report(lr_path)
        gb = parse_report(gb_path)

        rows.append(
            {
                "target": target,
                "decile": decile,
                "model": "LR",
                "R2": lr["R2"],
                "MAE": lr["MAE"],
                "RMSE": lr["RMSE"],
            }
        )
        rows.append(
            {
                "target": target,
                "decile": decile,
                "model": "GB",
                "R2": gb["R2"],
                "MAE": gb["MAE"],
                "RMSE": gb["RMSE"],
            }
        )
    return rows


def _fmt(x, width):
    if x is None:
        return "NA".rjust(width)
    return f"{x:.6f}".rjust(width)


def write_text_summary(rows_ret, rows_vol, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    def add_block(title, rows):
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)
        header = f"{'Decile':<6} | {'Model':<4} | {'R2':>10} | {'MAE':>12} | {'RMSE':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for decile in [f"ME{i}" for i in range(1, 11)]:
            for model in ["LR", "GB"]:
                row = next(r for r in rows if r["decile"] == decile and r["model"] == model)
                line = (
                    f"{decile:<6} | "
                    f"{model:<4} | "
                    f"{_fmt(row['R2'], 10)} | "
                    f"{_fmt(row['MAE'], 12)} | "
                    f"{_fmt(row['RMSE'], 12)}"
                )
                lines.append(line)
            lines.append("-" * len(header))
        lines.append("")

    add_block("Return forecast metrics (out-of-sample)", rows_ret)
    add_block("Volatility forecast metrics (out-of-sample)", rows_vol)

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    rows_ret = gather_rows(
        target="return",
        dir_lr=DIR_LR_RET,
        dir_gb=DIR_GB_RET,
        lr_tpl="{decile}_report_lr.txt",
        gb_tpl="{decile}_gb_walkforward_report.txt",
    )

    rows_vol = gather_rows(
        target="vol",
        dir_lr=DIR_LR_VOL,
        dir_gb=DIR_GB_VOL,
        lr_tpl="{decile}_vol_report_lr.txt",
        gb_tpl="{decile}_gb_vol_walkforward_report.txt",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "forecast_metrics_summary.txt"
    write_text_summary(rows_ret, rows_vol, out_path)


if __name__ == "__main__":
    main()
