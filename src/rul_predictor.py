from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from preprocessing import (
    INFORMATIVE_SENSORS,
    WINDOW_SIZES,
    DROP_COLS,
    assign_condition_labels,
    apply_scalers,
    compute_rolling_features,
    drop_low_variance_sensors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALARM_THRESHOLD: int = 30

# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_inference_artifacts(
    fd_id: str,
    model_name: str,
    artifacts_dir: str | Path,
) -> dict:
    """
    Load all artifacts needed for inference on one engine.

    Returns dict with keys: 'model', 'km', 'scalers', 'feature_cols'
    """
    base = Path(artifacts_dir) / fd_id
    return {
        "model": load(base / f"{model_name}.joblib"),
        "km": load(base / "kmeans.joblib"),
        "scalers": load(base / "scalers.joblib"),
        "feature_cols": json.loads((base / "feature_cols.json").read_text()),
    }


# ---------------------------------------------------------------------------
# Inference preprocessing
# ---------------------------------------------------------------------------


def preprocess_engine_for_inference(
    engine_df: pd.DataFrame,
    km: Any,
    scalers: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to raw cycles of one engine
    at inference time (no RUL computation needed).

    Steps: drop low-variance sensors → assign condition cluster →
    normalize with per-condition scaler → compute rolling features →
    return only feature_cols in training order.
    """
    df = drop_low_variance_sensors(engine_df.copy())
    df = assign_condition_labels(df, km)

    scale_cols = INFORMATIVE_SENSORS + ["op1", "op2"]
    df = apply_scalers(df, scalers, scale_cols)
    df = compute_rolling_features(df)

    # Ensure feature columns are in the correct order
    return df[feature_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def predict_rul(
    engine_df: pd.DataFrame,
    artifacts: dict,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict RUL for every cycle in engine_df.
    Returns array of length len(engine_df).
    """
    X = preprocess_engine_for_inference(
        engine_df, artifacts["km"], artifacts["scalers"], artifacts["feature_cols"]
    )
    preds = artifacts["model"].predict(X)
    if clip_negative:
        preds = np.clip(preds, 0, None)
    return preds.astype(float)


def predict_current_rul(
    engine_df: pd.DataFrame,
    artifacts: dict,
) -> float:
    """Predict RUL for only the LAST observed cycle of engine_df."""
    preds = predict_rul(engine_df, artifacts)
    return float(preds[-1])


# ---------------------------------------------------------------------------
# SHAP explainability
# ---------------------------------------------------------------------------


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_rows: int = 500,
):
    """
    Compute SHAP values using TreeExplainer (RF, XGB, LightGBM).
    Subsamples to max_rows for speed if X is large.
    Returns (shap.Explanation, base_values).
    """
    import shap

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    return shap_values, explainer.expected_value


def get_top_features(
    shap_values: np.ndarray,
    feature_cols: list[str],
    n: int = 8,
) -> list[str]:
    """Return names of top-n features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:n]
    return [feature_cols[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Alarm
# ---------------------------------------------------------------------------


def is_alarm(predicted_rul: float, threshold: int = ALARM_THRESHOLD) -> bool:
    """Return True if predicted RUL is at or below alarm threshold."""
    return predicted_rul <= threshold


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def plot_degradation_trajectory(
    engine_df: pd.DataFrame,
    predicted_ruls: np.ndarray,
    true_rul: float | None = None,
    threshold: int = ALARM_THRESHOLD,
) -> plt.Figure:
    """
    Plot predicted RUL trajectory over all cycles of one engine.

    - Blue line: predicted RUL at each cycle
    - Red dashed: alarm threshold
    - Green star: true RUL at last cycle (if provided)
    - Shaded red region: RUL < threshold
    """
    cycles = engine_df["cycle"].values
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(cycles, predicted_ruls, color="steelblue", lw=2, label="Predicted RUL")
    ax.axhline(threshold, color="red", linestyle="--", lw=1.5, label=f"Alarm (RUL={threshold})")
    ax.fill_between(cycles, 0, threshold, alpha=0.08, color="red")

    if true_rul is not None:
        ax.scatter(
            cycles[-1], true_rul,
            color="green", s=120, zorder=5, marker="*",
            label=f"True RUL = {true_rul:.0f}",
        )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Remaining Useful Life (cycles)")
    ax.set_title("Engine Degradation Trajectory")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_sensor_trends(
    engine_df: pd.DataFrame,
    sensors: list[str],
    n_cols: int = 2,
) -> plt.Figure:
    """
    Line charts for a list of sensors over the engine's cycles.
    Arranged in a grid with n_cols columns.
    """
    n_rows = int(np.ceil(len(sensors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    cycles = engine_df["cycle"].values
    for ax, sensor in zip(axes, sensors):
        if sensor in engine_df.columns:
            ax.plot(cycles, engine_df[sensor].values, lw=1.5, color="steelblue")
        ax.set_title(sensor)
        ax.set_xlabel("Cycle")
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(sensors):]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
