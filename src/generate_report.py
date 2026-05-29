"""
generate_report.py
------------------
Phase 10 - Generate a ydata-profiling HTML report for the cleaned Zomato dataset.

Usage
-----
    python src/generate_report.py              # full report (default)
    python src/generate_report.py --minimal    # faster minimal report
    python src/generate_report.py --explorative # deeper correlations & interactions

Output
------
    outputs/reports/zomato_profile.html
"""

import argparse
import pathlib
import sys
import time

import pandas as pd

ROOT        = pathlib.Path(__file__).resolve().parent.parent
CLEAN_CSV   = ROOT / "data" / "processed" / "zomato_clean.csv"
REPORT_DIR  = ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "zomato_profile.html"


def load_data() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        print(f"[report] ERROR: clean CSV not found at {CLEAN_CSV}")
        print("[report] Run the cleaning pipeline first:")
        print("         python -c \"from src.data_loader import load_raw_data; "
              "from src.cleaner import clean_data, engineer_features, save_clean_data; "
              "save_clean_data(engineer_features(clean_data(load_raw_data())))\"")
        sys.exit(1)

    df = pd.read_csv(CLEAN_CSV)
    # restore ordered categorical so profiling shows it correctly
    df["rating_bucket"] = pd.Categorical(
        df["rating_bucket"],
        categories=["Poor", "Average", "Good", "Excellent"],
        ordered=True,
    )
    print(f"[report] Loaded {len(df):,} rows x {df.shape[1]} cols from {CLEAN_CSV}")
    return df


def generate_report(df: pd.DataFrame, mode: str = "default") -> None:
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("[report] ERROR: ydata-profiling is not installed.")
        print("         Run:  pip install ydata-profiling")
        sys.exit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    config_kwargs = {}
    title = "Zomato Bengaluru - EDA Profile Report"

    if mode == "minimal":
        config_kwargs = {
            "minimal": True,
            "title": title,
        }
        print("[report] Generating MINIMAL report (fast)...")
    elif mode == "explorative":
        config_kwargs = {
            "title": title,
            "correlations": {
                "pearson":  {"calculate": True},
                "spearman": {"calculate": True},
                "kendall":  {"calculate": False},
                "phi_k":    {"calculate": False},
            },
            "interactions": {"continuous": True},
            "missing_diagrams": {
                "bar":    True,
                "matrix": True,
                "heatmap": True,
            },
        }
        print("[report] Generating EXPLORATIVE report (detailed, slower)...")
    else:
        config_kwargs = {
            "title": title,
            "correlations": {
                "pearson":  {"calculate": True},
                "spearman": {"calculate": True},
                "kendall":  {"calculate": False},
                "phi_k":    {"calculate": False},
            },
            "missing_diagrams": {
                "bar":    True,
                "matrix": True,
                "heatmap": False,
            },
            "duplicates": {"head": 10},
            "samples":    {"head": 10, "tail": 10},
        }
        print("[report] Generating DEFAULT report...")

    t0      = time.time()
    profile = ProfileReport(df, **config_kwargs)
    profile.to_file(REPORT_PATH)
    elapsed = time.time() - t0

    size_mb = REPORT_PATH.stat().st_size / (1024 * 1024)
    print(f"[report] Done in {elapsed:.1f}s  |  {size_mb:.2f} MB")
    print(f"[report] Report saved -> {REPORT_PATH}")
    print(f"[report] Open in browser: file:///{REPORT_PATH.as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ydata-profiling HTML report for Zomato dataset."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--minimal", action="store_true",
        help="Generate a fast minimal report (no correlations/interactions)."
    )
    group.add_argument(
        "--explorative", action="store_true",
        help="Generate a deep explorative report with interactions."
    )
    args = parser.parse_args()

    mode = "minimal" if args.minimal else ("explorative" if args.explorative else "default")

    df = load_data()
    generate_report(df, mode=mode)


if __name__ == "__main__":
    main()
