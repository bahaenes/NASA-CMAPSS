"""
Turbofan Engine RUL Predictor — Streamlit Dashboard
Run with: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add src/ to path so imports work regardless of working directory
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from preprocessing import load_raw, INFORMATIVE_SENSORS
from rul_predictor import (
    load_inference_artifacts,
    predict_rul,
    compute_shap_values,
    get_top_features,
    is_alarm,
    plot_degradation_trajectory,
    ALARM_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROC = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

DS_INFO = {
    "FD001": {"n_engines": 100,  "n_conditions": 1, "fault": "HPC Degradation"},
    "FD002": {"n_engines": 259,  "n_conditions": 6, "fault": "HPC Degradation"},
    "FD003": {"n_engines": 100,  "n_conditions": 1, "fault": "HPC + Fan Degradation"},
    "FD004": {"n_engines": 248,  "n_conditions": 6, "fault": "HPC + Fan Degradation"},
}

AVAILABLE_MODELS = ["xgboost", "lightgbm", "random_forest", "ridge", "linear_regression"]

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Turbofan RUL Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading model artifacts...")
def get_artifacts(fd_id: str, model_name: str) -> dict | None:
    try:
        return load_inference_artifacts(fd_id, model_name, MODELS_DIR)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Loading engine data...")
def get_test_data(fd_id: str) -> pd.DataFrame:
    return load_raw(DATA_RAW, fd_id, "test")


@st.cache_data(show_spinner=False)
def get_true_rul(fd_id: str) -> pd.Series | None:
    from preprocessing import load_rul_labels
    try:
        return load_rul_labels(DATA_RAW, fd_id)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_model_metrics(fd_id: str, model_name: str) -> dict | None:
    path = MODELS_DIR / fd_id / f"{model_name}_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("✈️ Engine Configuration")
    st.divider()

    fd_id = st.selectbox(
        "Dataset",
        options=list(DS_INFO.keys()),
        help="FD001/FD003: single operating condition\nFD002/FD004: 6 operating conditions",
    )
    info = DS_INFO[fd_id]
    max_engines = info["n_engines"]

    unit_id = st.number_input(
        "Engine Unit ID",
        min_value=1,
        max_value=max_engines,
        value=1,
        step=1,
    )

    # Filter available models to those that have been trained
    available = [
        m for m in AVAILABLE_MODELS
        if (MODELS_DIR / fd_id / f"{m}.joblib").exists()
    ]
    if not available:
        available = AVAILABLE_MODELS  # show all if none trained yet

    model_name = st.selectbox("Model", options=available)

    st.divider()
    st.caption(f"**Operating conditions:** {info['n_conditions']}")
    st.caption(f"**Fault mode:** {info['fault']}")
    st.caption(f"**Test engines:** {max_engines}")
    st.divider()
    st.caption(f"Alarm threshold: RUL < {ALARM_THRESHOLD} cycles")

# ---------------------------------------------------------------------------
# Load artifacts and data
# ---------------------------------------------------------------------------

artifacts = get_artifacts(fd_id, model_name)
test_df = get_test_data(fd_id)
true_ruls = get_true_rul(fd_id)

engine_df = test_df[test_df["unit_id"] == unit_id].copy()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Turbofan Engine Predictive Maintenance")
st.markdown(
    f"**Dataset:** {fd_id} &nbsp;|&nbsp; "
    f"**Engine:** {unit_id} &nbsp;|&nbsp; "
    f"**Model:** {model_name}"
)

if artifacts is None:
    st.error(
        f"No trained model found for {fd_id} / {model_name}. "
        "Please run `notebooks/03_modeling.ipynb` first to train and save models."
    )
    st.stop()

if len(engine_df) == 0:
    st.error(f"No test data found for engine {unit_id} in {fd_id}.")
    st.stop()

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

with st.spinner("Computing predictions..."):
    try:
        predicted_ruls = predict_rul(engine_df, artifacts)
        current_rul = float(predicted_ruls[-1])
        alarm = is_alarm(current_rul)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

true_rul_val = float(true_ruls[unit_id]) if true_ruls is not None else None

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Predicted RUL", f"{current_rul:.0f} cycles")
col2.metric("Last Observed Cycle", int(engine_df["cycle"].max()))
col3.metric("Cycles in Test Sequence", len(engine_df))
if true_rul_val is not None:
    error = current_rul - true_rul_val
    col4.metric("True RUL", f"{true_rul_val:.0f}", delta=f"{error:+.0f} pred error")

# Alarm banner
if alarm:
    st.error(
        f"⚠️ ALARM: Predicted RUL = {current_rul:.0f} cycles — below threshold of {ALARM_THRESHOLD}. "
        "Schedule maintenance immediately."
    )
else:
    st.success(
        f"Engine operating within safe range. "
        f"Predicted RUL = {current_rul:.0f} cycles (threshold: {ALARM_THRESHOLD})."
    )

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Degradation Trajectory",
    "📡 Sensor Trends",
    "🔍 Feature Importance (SHAP)",
    "📊 Model Metrics",
])

# --- Tab 1: Degradation trajectory ---
with tab1:
    st.subheader("RUL Degradation Trajectory")
    fig = plot_degradation_trajectory(engine_df, predicted_ruls, true_rul_val)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"- **Blue line**: predicted RUL at each observed cycle  \n"
        f"- **Red dashed**: alarm threshold (RUL = {ALARM_THRESHOLD})  \n"
        f"- **Green star**: true RUL at last cycle (from RUL label file)"
        if true_rul_val is not None else
        f"- **Blue line**: predicted RUL at each observed cycle  \n"
        f"- **Red dashed**: alarm threshold (RUL = {ALARM_THRESHOLD})"
    )

# --- Tab 2: Sensor trends ---
with tab2:
    st.subheader("Sensor Trends")

    # Try to get SHAP-guided top sensors; fall back to default list
    shap_vals = None
    try:
        with st.spinner("Computing SHAP values for sensor selection..."):
            X_proc = __import__("rul_predictor").preprocess_engine_for_inference(
                engine_df, artifacts["km"], artifacts["scalers"], artifacts["feature_cols"]
            )
            shap_vals, _ = compute_shap_values(artifacts["model"], X_proc, max_rows=200)
            top_raw_sensors = [
                s for s in get_top_features(shap_vals.values, artifacts["feature_cols"], n=20)
                if s in engine_df.columns
            ][:6]
    except Exception:
        top_raw_sensors = []

    if not top_raw_sensors:
        top_raw_sensors = INFORMATIVE_SENSORS[:6]

    st.caption(f"Showing top sensors: {', '.join(top_raw_sensors)}")

    cols = st.columns(2)
    for i, sensor in enumerate(top_raw_sensors):
        with cols[i % 2]:
            fig2, ax = plt.subplots(figsize=(6, 3))
            ax.plot(engine_df["cycle"].values, engine_df[sensor].values,
                    lw=1.5, color="steelblue")
            ax.set_title(sensor)
            ax.set_xlabel("Cycle")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

# --- Tab 3: SHAP feature importance ---
with tab3:
    st.subheader("SHAP Feature Importance")
    try:
        import shap

        if shap_vals is None:
            with st.spinner("Computing SHAP values..."):
                X_proc = __import__("rul_predictor").preprocess_engine_for_inference(
                    engine_df, artifacts["km"], artifacts["scalers"], artifacts["feature_cols"]
                )
                shap_vals, _ = compute_shap_values(artifacts["model"], X_proc, max_rows=200)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_vals, max_display=15, show=False, ax=ax3)
        ax3.set_title(f"{fd_id} / {model_name} — Mean |SHAP| Feature Importance")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.caption(
            "SHAP values show how much each feature pushes the RUL prediction "
            "higher (positive) or lower (negative) relative to the model's average prediction."
        )

    except ImportError:
        st.warning("SHAP not installed. Run `pip install shap` to enable explainability.")
    except Exception as e:
        st.warning(f"SHAP computation failed for this model type: {e}")

# --- Tab 4: Model metrics ---
with tab4:
    st.subheader("Saved Model Metrics")
    metrics = load_model_metrics(fd_id, model_name)

    if metrics:
        m = metrics.get("metrics", {})
        if m:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("RMSE", f"{m.get('rmse', 'N/A'):.2f}" if isinstance(m.get("rmse"), float) else "N/A")
            mcol2.metric("MAE",  f"{m.get('mae', 'N/A'):.2f}"  if isinstance(m.get("mae"), float) else "N/A")
            mcol3.metric("NASA Score", f"{m.get('nasa_score', 'N/A'):.0f}" if isinstance(m.get("nasa_score"), float) else "N/A")
        st.caption(f"Trained on {fd_id} | Saved at {metrics.get('saved_at', 'unknown')}")
    else:
        st.info(
            "No metrics file found. Train models in `notebooks/03_modeling.ipynb` first.\n\n"
            "Expected RMSE on FD001 test set:\n"
            "- Linear Regression: ~23\n"
            "- Random Forest: ~16\n"
            "- XGBoost / LightGBM: ~14"
        )

    # Show metrics for all available models in this dataset
    st.divider()
    st.subheader("All Models Comparison")
    rows = []
    for m_name in AVAILABLE_MODELS:
        m_meta = load_model_metrics(fd_id, m_name)
        if m_meta and m_meta.get("metrics"):
            rows.append({"Model": m_name, **m_meta["metrics"]})
    if rows:
        df_metrics = pd.DataFrame(rows).set_index("Model").sort_values("rmse")
        st.dataframe(df_metrics.style.highlight_min(axis=0, color="lightgreen"), use_container_width=True)
    else:
        st.info("Train models first to see comparison.")
