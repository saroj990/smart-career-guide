#!/usr/bin/env python3
"""Run Day 7 model benchmark from project root (no Jupyter required)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_evaluation import (
    benchmark_classifiers,
    pick_best_model_name,
    save_day07_artifacts,
)
from src.preprocessing import load_cleaned_table, prepare_train_test_bundle, save_artifacts


def main() -> None:
    df = load_cleaned_table(ROOT)
    bundle = prepare_train_test_bundle(df, test_size=0.2, random_state=42)
    save_artifacts(bundle, ROOT)
    comparison, fitted, _ = benchmark_classifiers(
        bundle["X_train"],
        bundle["X_test"],
        bundle["y_train"],
        bundle["y_test"],
        bundle["career_classes"],
        cv=5,
    )
    best = pick_best_model_name(comparison)
    paths = save_day07_artifacts(
        comparison, fitted, bundle["career_classes"], ROOT, best_name=best
    )
    print("Best model:", best)
    print(comparison.to_string(index=False))
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
