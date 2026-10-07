#!/usr/bin/env python3
"""Train Day 8 ANN from project root."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split

from src.ann_models import build_ann, evaluate_ann, save_day08_artifacts, train_ann
from src.preprocessing import load_cleaned_table, prepare_train_test_bundle, save_artifacts


def main() -> None:
    df = load_cleaned_table(ROOT)
    bundle = prepare_train_test_bundle(df, test_size=0.2, random_state=42)
    save_artifacts(bundle, ROOT)
    X_train, y_train = bundle["X_train"], bundle["y_train"]
    X_test, y_test = bundle["X_test"], bundle["y_test"]
    classes = bundle["career_classes"]

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    model = build_ann(X_tr.shape[1], len(classes))
    history = train_ann(model, X_tr, y_tr, X_val, y_val)
    metrics = evaluate_ann(model, X_test, y_test, classes)
    paths = save_day08_artifacts(model, history, metrics, classes, ROOT)
    print("ANN test F1 macro:", metrics["f1_macro"])
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
