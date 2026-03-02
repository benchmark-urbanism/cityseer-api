#!/usr/bin/env python
"""
run_all.py - Run the full sampling analysis pipeline in order.

Usage:
    python run_all.py           # Run all scripts (uses cached data where available)
    python run_all.py --force   # Force regeneration of all cached data
    python run_all.py --from 2  # Resume from script 02 onwards
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

SCRIPTS = [
    ("00_generate_cache.py", "Generate synthetic network sampling data", True),
    ("01_analyse_synthetic.py", "Analyse synthetic data and generate figures", False),
    ("02_validate_gla.py", "Validate on Greater London network", True),
    ("03_validate_madrid.py", "Validate on Greater Madrid network", True),
    ("04_figures_validation.py", "Generate validation figures for GLA and Madrid", False),
    ("05_generate_macros.py", "Generate LaTeX macros, tables, and practical guide figure", False),
    ("06_figures_spatial.py", "Generate spatial error figures from per-node caches", False),
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
        help="Start from script N (e.g. --from 2 to start at 02_validate_gla.py)",
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
