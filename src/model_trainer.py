from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 asymmetric scoring function.

    d = y_pred - y_true
      d < 0  (early prediction): S += exp(-d/13) - 1  [less penalized]
      d >= 0 (late prediction):  S += exp(d/10)  - 1  [more penalized]

    Lower is better. Penalizes predicting MORE remaining life than actual,
    which is the dangerous failure mode in maintenance planning.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict[str, float]:
    """
    Compute RMSE, MAE, and NASA score.
    Clips negative RUL predictions to 0 before scoring.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    ns = nasa_score(y_true, y_pred_clipped)
    if model_name:
        print(f"  {model_name:<20} RMSE={rmse:.2f}  MAE={mae:.2f}  NASA={ns:.1f}")
    return {"rmse": rmse, "mae": mae, "nasa_score": ns}


# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------


def build_model_zoo() -> dict[str, Any]:
    """
    Return a dict of {name: unfitted estimator} with tuned hyperparameters.

    XGBoost and LightGBM are imported lazily so the module loads even if
    they are not installed (notebooks will fail gracefully).
    """
    zoo: dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=42,
        ),
    }

    try:
        from xgboost import XGBRegressor

        zoo["xgboost"] = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric="rmse",
        )
    except ImportError:
        print("XGBoost not installed — skipping.")

    try:
        from lightgbm import LGBMRegressor

        zoo["lightgbm"] = LGBMRegressor(
            n_estimators=500,
            num_leaves=63,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        print("LightGBM not installed — skipping.")

    return zoo


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def _supports_early_stopping(model) -> bool:
    model_type = type(model).__name__
    return model_type in ("XGBRegressor", "LGBMRegressor")


def cross_validate_by_engine(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    GroupKFold CV where each group is one engine unit_id.
    Guarantees no engine cycles appear in both train and validation folds.

    For XGBoost/LightGBM, passes eval_set for early stopping.
    Returns {metric: [fold_scores]}.
    """
    import copy

    gkf = GroupKFold(n_splits=n_splits)
    fold_results: dict[str, list[float]] = {"rmse": [], "mae": [], "nasa_score": []}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        m = copy.deepcopy(model)

        if _supports_early_stopping(m):
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            m.fit(X_tr, y_tr)

        y_pred = m.predict(X_val)
        metrics = evaluate_model(y_val.values, y_pred)
        for k, v in metrics.items():
            fold_results[k].append(v)

        if verbose:
            print(
                f"  Fold {fold_idx + 1}/{n_splits}  "
                f"RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}"
            )

    if verbose:
        print(
            f"  CV mean RMSE={np.mean(fold_results['rmse']):.2f} "
            f"± {np.std(fold_results['rmse']):.2f}"
        )

    return fold_results


# ---------------------------------------------------------------------------
# Final training
# ---------------------------------------------------------------------------


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Any:
    """
    Train on the full training set.
    If the model supports early stopping and validation data is provided,
    use it as the stopping criterion.
    """
    if _supports_early_stopping(model) and X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: Any,
    model_name: str,
    fd_id: str,
    output_dir: str | Path,
    metrics: dict | None = None,
    feature_cols: list[str] | None = None,
) -> Path:
    """
    Save model to models/{fd_id}/{model_name}.joblib.
    Also writes a _metrics.json file with evaluation results.
    """
    out = Path(output_dir) / fd_id
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / f"{model_name}.joblib"
    dump(model, model_path)

    meta = {
        "model_name": model_name,
        "fd_id": fd_id,
        "saved_at": datetime.utcnow().isoformat(),
        "metrics": metrics or {},
        "feature_cols": feature_cols or [],
    }
    (out / f"{model_name}_metrics.json").write_text(json.dumps(meta, indent=2))
    return model_path


def load_model(model_name: str, fd_id: str, model_dir: str | Path) -> Any:
    """Load a joblib model from models/{fd_id}/{model_name}.joblib."""
    path = Path(model_dir) / fd_id / f"{model_name}.joblib"
    return load(path)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------


def build_results_table(
    results: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Convert {model_name: {metric: value}} to a DataFrame sorted by RMSE.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({"model": name, **metrics})
    df = pd.DataFrame(rows).set_index("model")
    return df.sort_values("rmse")


# ---------------------------------------------------------------------------
# Optional LSTM
# ---------------------------------------------------------------------------


def build_lstm_model(
    sequence_length: int = 30,
    n_features: int = 100,
    units: list[int] | None = None,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
):
    """
    Two-layer stacked LSTM with dropout.

    Architecture:
      Input(shape=(sequence_length, n_features))
      -> LSTM(units[0], return_sequences=True) -> Dropout
      -> LSTM(units[1], return_sequences=False) -> Dropout
      -> Dense(1, activation='relu')   # relu prevents negative RUL

    sequence_length=30 matches the largest rolling window and is
    the standard choice in CMAPSS LSTM literature.
    """
    if units is None:
        units = [64, 32]

    try:
        from tensorflow import keras
    except ImportError:
        raise ImportError("TensorFlow is required for the LSTM model.")

    model = keras.Sequential([
        keras.layers.Input(shape=(sequence_length, n_features)),
        keras.layers.LSTM(units[0], return_sequences=True),
        keras.layers.Dropout(dropout),
        keras.layers.LSTM(units[1], return_sequences=False),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1, activation="relu"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def prepare_lstm_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    sequence_length: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert tabular features into (N, sequence_length, n_features) tensor.

    Creates overlapping windows per engine. The label for each window is
    the RUL at the last row of that window.
    Short sequences are zero-padded on the left (left-padding).
    """
    X_seqs, y_seqs = [], []

    for unit_id in groups.unique():
        mask = groups == unit_id
        X_engine = X[mask].values
        y_engine = y[mask].values
        n = len(X_engine)

        for i in range(n):
            end = i + 1
            start = end - sequence_length
            if start < 0:
                pad_len = -start
                seq = np.vstack([
                    np.zeros((pad_len, X_engine.shape[1])),
                    X_engine[:end],
                ])
            else:
                seq = X_engine[start:end]
            X_seqs.append(seq)
            y_seqs.append(y_engine[i])

    return np.array(X_seqs, dtype=np.float32), np.array(y_seqs, dtype=np.float32)
