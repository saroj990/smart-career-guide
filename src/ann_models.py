"""
Day 8 ANN (Keras) — notebooks/day08/day08_ann.ipynb.

  from src.ann_models import build_ann, train_ann, evaluate_ann, save_day08_artifacts
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.classification import evaluate_multiclass


def build_ann(
    n_features: int,
    n_classes: int,
    hidden_units: tuple[int, ...] = (64, 32),
    dropout: tuple[float, ...] = (0.3, 0.2),
) -> Any:
    """Feed-forward network: ReLU hidden layers + softmax output."""
    from tensorflow import keras
    from tensorflow.keras import layers

    if len(dropout) < len(hidden_units):
        dropout = tuple(list(dropout) + [0.0] * (len(hidden_units) - len(dropout)))

    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if dropout[i] > 0:
            x = layers.Dropout(dropout[i], name=f"dropout_{i}")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="career_probs")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="career_ann")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 32,
    patience: int = 10,
) -> Any:
    """Fit with early stopping on validation loss."""
    from tensorflow.keras.callbacks import EarlyStopping

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    return history


def _sklearn_like_wrapper(model: Any):
    """Minimal adapter so evaluate_multiclass can call predict / predict_proba."""

    class _Wrap:
        def predict(self, X: np.ndarray) -> np.ndarray:
            probs = model.predict(X, verbose=0)
            return np.argmax(probs, axis=1)

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return model.predict(X, verbose=0)

    return _Wrap()


def evaluate_ann(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    wrapper = _sklearn_like_wrapper(model)
    return evaluate_multiclass(wrapper, X_test, y_test, class_names)


def plot_training_history(history: Any, title: str = "ANN training history") -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    h = history.history
    axes[0].plot(h["loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(h["accuracy"], label="train")
    axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def save_day08_artifacts(
    model: Any,
    history: Any,
    metrics: dict[str, Any],
    class_names: list[str],
    project_root: Path,
) -> dict[str, Path]:
    figures_dir = project_root / "outputs" / "figures"
    metrics_dir = project_root / "outputs" / "metrics"
    models_dir = project_root / "models"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_training_history(history)
    fig_path = figures_dir / "day08_ann_training_history.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "model": "keras_ann",
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "n_classes": len(class_names),
        "career_classes": class_names,
    }
    metrics_path = metrics_dir / "day08_ann_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = metrics_dir / "day08_ann_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    model_path = models_dir / "day08_ann.keras"
    model.save(model_path)

    return {
        "training_figure": fig_path,
        "metrics": metrics_path,
        "report": report_path,
        "model": model_path,
    }


def load_day08_ann(project_root: Path) -> Any:
    from tensorflow import keras

    path = project_root / "models" / "day08_ann.keras"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run Day 8 notebook or train_ann pipeline.")
    return keras.models.load_model(path)


def ann_metrics_only(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    wrapper = _sklearn_like_wrapper(model)
    y_pred = wrapper.predict(X_test)
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }
