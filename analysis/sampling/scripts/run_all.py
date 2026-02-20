#!/usr/bin/env python
"""
run_all.py - Run the full sampling analysis pipeline in order.

Usage:
    python run_all.py           # Run all scripts (uses cached data where available)
    python run_all.py --force   # Force regeneration of all cached data
    python run_all.py --from 3  # Resume from script 03 onwards
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SCRIPTS = [
    ("00_generate_cache.py", "Generate synthetic network sampling data", True),
    ("01_fit_rank_model.py", "Analyse synthetic data and generate figures", False),
    ("02_fit_error_model.py", "Validate Hoeffding/EW bound on synthetic data", False),
    ("03_validate_gla.py", "Validate model on Greater London network", True),
    ("04_validate_madrid.py", "Validate model on Greater Madrid network", True),
    ("05_practical_guide.py", "Generate practical guidance figures/tables", False),
    ("06_generate_macros.py", "Generate LaTeX macros from results", False),
    ("07_hoeffding_model_figure.py", "Generate Hoeffding model figure", False),
]


def main():
    parser = argparse.ArgumentParser(description="Run the full sampling analysis pipeline")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all cached data")
    parser.add_argument(
        "--from",
        type=int,
        default=0,
        dest="start_from",
        metavar="N",
        help="Start from script N (e.g. --from 3 to start at 03_validate_gla.py)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SAMPLING ANALYSIS PIPELINE")
    print("=" * 70)
    if args.force:
        print("  Mode: FORCE (regenerating all cached data)")
    if args.start_from > 0:
        print(f"  Starting from script {args.start_from:02d}")
    print()

    total_start = time.time()
    failed = []

    for filename, description, supports_force in SCRIPTS:
        script_num = int(filename[:2])
        if script_num < args.start_from:
            continue

        print(f"\n{'=' * 70}")
        print(f"[{filename}] {description}")
        print("=" * 70)

        cmd = [sys.executable, str(SCRIPT_DIR / filename)]
        if args.force and supports_force:
            cmd.append("--force")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nFAILED: {filename} (exit code {result.returncode}, {elapsed:.1f}s)")
            failed.append(filename)
            break
        else:
            print(f"\nDone: {filename} ({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_elapsed:.1f}s")

    if failed:
        print(f"  FAILED: {', '.join(failed)}")
        return 1

    print("  All scripts completed successfully.")
    return 0


if __name__ == "__main__":
    exit(main())
