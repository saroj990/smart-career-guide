#!/usr/bin/env python3
"""Build Day 9 comparison table and production bundle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_comparison import build_comparison_table, save_production_model
from src.preprocessing import load_cleaned_table, prepare_train_test_bundle


def main() -> None:
    bundle = prepare_train_test_bundle(load_cleaned_table(ROOT))
    comparison = build_comparison_table(ROOT)
    paths = save_production_model(
        comparison,
        ROOT,
        X_train=bundle["X_train"],
        y_train=bundle["y_train"],
        career_classes=bundle["career_classes"],
    )
    print(comparison[["model", "test_accuracy", "test_f1_macro", "family"]].to_string(index=False))
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
