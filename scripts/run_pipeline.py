#!/usr/bin/env python3
"""
Run Days 7–14 artifacts in order (requires prior Days 1–3 data or raw CSV).

Usage (from project root):
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --from-day 8
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    (7, ROOT / "scripts" / "run_day07_benchmark.py"),
    (8, ROOT / "scripts" / "run_day08_ann.py"),
    (9, ROOT / "scripts" / "run_day09_comparison.py"),
]


def _run_day14_exports() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.tableau_export import demo_profile_exports, export_tableau_tables
    from src.testing import save_test_report

    export_tableau_tables(ROOT)
    demo_profile_exports(ROOT)
    save_test_report(ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-day", type=int, default=7)
    args = parser.parse_args()
    for day, script in SCRIPTS:
        if day < args.from_day:
            continue
        print(f"\n=== Day {day} ===")
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
    if args.from_day <= 14:
        print("\n=== Day 14 exports & tests ===")
        _run_day14_exports()
    print("\nPipeline complete. Launch app: streamlit run app.py")


if __name__ == "__main__":
    main()
