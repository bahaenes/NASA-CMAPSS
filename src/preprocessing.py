from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLUMNS: list[str] = [
    "unit_id", "cycle",
    "op1", "op2", "op3",
    "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10",
    "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21",
]

# Zero / near-zero variance in all datasets — confirmed from data exploration
DROP_COLS: list[str] = ["s1", "s5", "s6", "s10", "s16", "s18", "s19", "op3"]

INFORMATIVE_SENSORS: list[str] = [
    "s2", "s3", "s4", "s7", "s8", "s9",
    "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21",
]

RUL_CAP: int = 125
WINDOW_SIZES: list[int] = [5, 10, 30]

# Number of operating conditions per dataset
N_CONDITIONS: dict[str, int] = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_raw(
    data_dir: str | Path,
    fd_id: Literal["FD001", "FD002", "FD003", "FD004"],
    split: Literal["train", "test"],
) -> pd.DataFrame:
    """Load a raw CMAPSS text file, assign column names, add dataset tag."""
    path = Path(data_dir) / f"{split}_{fd_id}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    df.insert(0, "dataset", fd_id)
    return df


def load_rul_labels(
    data_dir: str | Path,
    fd_id: Literal["FD001", "FD002", "FD003", "FD004"],
) -> pd.Series:
    """
    Load RUL_FDxxx.txt.
    Returns a Series indexed 1..N mapping test engine number -> true RUL
    at its last observed cycle.
    """
    path = Path(data_dir) / f"RUL_{fd_id}.txt"
    s = pd.read_csv(path, header=None, names=["true_rul"]).squeeze()
    s.index = s.index + 1  # 1-based engine IDs
    s.index.name = "unit_id"
    return s


# ---------------------------------------------------------------------------
# RUL computation
# ---------------------------------------------------------------------------


def compute_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Add a piecewise-linear capped RUL column to training data.

    RUL = min(max_cycle_for_engine - current_cycle, cap)

    The cap reflects that the linear degradation model is meaningful only
    in the last ~125 cycles. Earlier rows are set to 125 (healthy zone).
    """
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["rul"] = (df["max_cycle"] - df["cycle"]).clip(upper=cap)
    df.drop(columns="max_cycle", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------


def drop_low_variance_sensors(
    df: pd.DataFrame,
    cols_to_drop: list[str] = DROP_COLS,
) -> pd.DataFrame:
    """Drop known zero/near-zero variance columns (safe if already absent)."""
    present = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=present)


# ---------------------------------------------------------------------------
# Operating condition clustering
# ---------------------------------------------------------------------------


def fit_condition_clusters(df: pd.DataFrame, n_clusters: int = 6) -> KMeans:
    """
    Fit KMeans on (op1, op2) to identify operating conditions.
    Pass n_clusters=1 for FD001/FD003 (single condition).
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(df[["op1", "op2"]].values)
    return km


def assign_condition_labels(df: pd.DataFrame, km: KMeans) -> pd.DataFrame:
    """Append 'flight_condition' column (int) using a fitted KMeans."""
    df = df.copy()
    df["flight_condition"] = km.predict(df[["op1", "op2"]].values)
    return df


# ---------------------------------------------------------------------------
# Per-condition normalization
# ---------------------------------------------------------------------------


def fit_scalers_per_condition(
    df: pd.DataFrame,
    feature_cols: list[str],
    condition_col: str = "flight_condition",
) -> dict[int, MinMaxScaler]:
    """
    Fit one MinMaxScaler per condition label on training data only.
    Returns {condition_id: fitted_scaler}.
    """
    scalers: dict[int, MinMaxScaler] = {}
    for cond, group in df.groupby(condition_col):
        scaler = MinMaxScaler()
        scaler.fit(group[feature_cols].values)
        scalers[int(cond)] = scaler
    return scalers


def apply_scalers(
    df: pd.DataFrame,
    scalers: dict[int, MinMaxScaler],
    feature_cols: list[str],
    condition_col: str = "flight_condition",
) -> pd.DataFrame:
    """
    Normalize feature_cols in-place using per-condition scalers.
    Raises KeyError if a condition is not found in scalers.
    """
    df = df.copy()
    for cond, group_idx in df.groupby(condition_col).groups.items():
        scaler = scalers[int(cond)]
        df.loc[group_idx, feature_cols] = scaler.transform(
            df.loc[group_idx, feature_cols].values
        )
    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


def compute_rolling_features(
    df: pd.DataFrame,
    sensors: list[str] = INFORMATIVE_SENSORS,
    windows: list[int] = WINDOW_SIZES,
) -> pd.DataFrame:
    """
    Compute per-engine rolling mean and std for each sensor and window size.

    Naming: {sensor}_roll{window}_{mean|std}

    NaN at the start of each engine's history (where the window cannot be
    filled) is resolved with ffill then bfill per engine — NOT global mean.
    """
    df = df.copy()
    new_cols: dict[str, pd.Series] = {}

    for window in windows:
        for sensor in sensors:
            for stat in ("mean", "std"):
                col_name = f"{sensor}_roll{window}_{stat}"
                rolled = (
                    df.groupby("unit_id")[sensor]
                    .transform(lambda s, w=window, st=stat: getattr(s.rolling(w, min_periods=1), st)())
                )
                new_cols[col_name] = rolled

    rolling_df = pd.DataFrame(new_cols, index=df.index)

    # Fill any remaining NaN per engine with ffill then bfill
    for col in rolling_df.columns:
        rolling_df[col] = (
            rolling_df.groupby(df["unit_id"])[col]
            .transform(lambda s: s.ffill().bfill())
        )

    return pd.concat([df, rolling_df], axis=1)


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------


def split_train_val_by_engine(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at the engine level to prevent time-series leakage.
    No engine appears in both train and val.
    """
    rng = np.random.default_rng(random_state)
    unit_ids = df["unit_id"].unique()
    n_val = max(1, int(len(unit_ids) * val_fraction))
    val_ids = rng.choice(unit_ids, size=n_val, replace=False)
    val_mask = df["unit_id"].isin(val_ids)
    return df[~val_mask].copy(), df[val_mask].copy()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_processed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    output_dir: str | Path,
    fd_id: str,
) -> None:
    """Save processed splits and feature column list to parquet files."""
    out = Path(output_dir) / fd_id
    out.mkdir(parents=True, exist_ok=True)

    label_col = "rul"

    train_df[feature_cols].to_parquet(out / "train_features.parquet", index=False)
    train_df[[label_col]].to_parquet(out / "train_labels.parquet", index=False)

    val_df[feature_cols].to_parquet(out / "val_features.parquet", index=False)
    val_df[[label_col]].to_parquet(out / "val_labels.parquet", index=False)

    # test has no RUL — use load_rul_labels separately
    test_df[feature_cols].to_parquet(out / "test_features.parquet", index=False)

    # Save unit_id + cycle for later alignment
    train_df[["unit_id", "cycle"]].to_parquet(out / "train_meta.parquet", index=False)
    val_df[["unit_id", "cycle"]].to_parquet(out / "val_meta.parquet", index=False)
    test_df[["unit_id", "cycle"]].to_parquet(out / "test_meta.parquet", index=False)

    (out / "feature_cols.json").write_text(json.dumps(feature_cols))


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    fd_id: Literal["FD001", "FD002", "FD003", "FD004"],
    n_conditions: int | None = None,
    rul_cap: int = RUL_CAP,
    val_fraction: float = 0.2,
) -> dict:
    """
    Full preprocessing pipeline for one CMAPSS dataset variant.

    Steps:
      1. Load train + test raw data
      2. Drop low-variance sensors
      3. Compute RUL for train (piecewise-linear capped)
      4. Fit condition clusters on train op settings
      5. Assign condition labels to train + test
      6. Fit per-condition scalers on train
      7. Apply scalers to train + test
      8. Compute rolling features for train + test
      9. Build feature column list
     10. Split train into train/val by engine
     11. Save parquet files + artifacts

    Returns dict with fitted artifacts: km, scalers, feature_cols.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    models_dir = output_dir.parent / "models" / fd_id
    models_dir.mkdir(parents=True, exist_ok=True)

    if n_conditions is None:
        n_conditions = N_CONDITIONS[fd_id]

    print(f"\n{'='*50}")
    print(f"Processing {fd_id} | {n_conditions} operating condition(s)")
    print(f"{'='*50}")

    # 1. Load
    train_raw = load_raw(raw_dir, fd_id, "train")
    test_raw = load_raw(raw_dir, fd_id, "test")
    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

    # 2. Drop low-variance sensors
    train = drop_low_variance_sensors(train_raw)
    test = drop_low_variance_sensors(test_raw)

    # 3. Compute RUL for train
    train = compute_train_rul(train, cap=rul_cap)

    # 4. Fit condition clusters
    km = fit_condition_clusters(train, n_clusters=n_conditions)
    dump(km, models_dir / "kmeans.joblib")

    # 5. Assign condition labels
    train = assign_condition_labels(train, km)
    test = assign_condition_labels(test, km)

    # 6. Fit scalers on train sensors + op cols
    scale_cols = INFORMATIVE_SENSORS + ["op1", "op2"]
    scalers = fit_scalers_per_condition(train, scale_cols)
    dump(scalers, models_dir / "scalers.joblib")

    # 7. Apply scalers
    train = apply_scalers(train, scalers, scale_cols)
    test = apply_scalers(test, scalers, scale_cols)

    # 8. Rolling features
    train = compute_rolling_features(train)
    test = compute_rolling_features(test)

    # 9. Feature column list
    rolling_cols = [
        f"{s}_roll{w}_{stat}"
        for w in WINDOW_SIZES
        for s in INFORMATIVE_SENSORS
        for stat in ("mean", "std")
    ]
    feature_cols = INFORMATIVE_SENSORS + rolling_cols + ["op1", "op2"]
    print(f"Total features: {len(feature_cols)}")

    # 10. Train/val split
    train_split, val_split = split_train_val_by_engine(train, val_fraction=val_fraction)
    print(
        f"Train engines: {train_split['unit_id'].nunique()}, "
        f"Val engines: {val_split['unit_id'].nunique()}, "
        f"Test engines: {test['unit_id'].nunique()}"
    )

    # 11. Save
    save_processed(train_split, val_split, test, feature_cols, output_dir, fd_id)
    print(f"Saved to {output_dir / fd_id}/")

    return {"km": km, "scalers": scalers, "feature_cols": feature_cols}
