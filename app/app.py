import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="wide"
)

# ============================================================
# LOAD MODEL BUNDLE
# ============================================================
@st.cache_resource
def load_model():
    with open(r"D:\crop_recommendation_ieee\models\model_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle       = load_model()
model        = bundle["model"]
le           = bundle["label_encoder"]
feature_names = bundle["feature_names"]
stats        = bundle["feature_stats"]

# ============================================================
# FEATURE ENGINEERING — must match training exactly
# ============================================================
def engineer_features(N, P, K, temperature, humidity, ph, rainfall, stats):
    feats = {}

    # Base features
    feats["N"]           = N
    feats["P"]           = P
    feats["K"]           = K
    feats["temperature"] = temperature
    feats["humidity"]    = humidity
    feats["ph"]          = ph
    feats["rainfall"]    = rainfall

    # Nutrient ratios
    feats["N_P_ratio"]   = N / (P + 1e-6)
    feats["N_K_ratio"]   = N / (K + 1e-6)
    feats["P_K_ratio"]   = P / (K + 1e-6)
    feats["NPK_total"]   = N + P + K
    feats["NPK_balance"] = N / (N + P + K + 1e-6)

    # Climate interactions
    feats["temp_humidity"]     = temperature * humidity
    feats["rainfall_humidity"] = rainfall    * humidity
    feats["temp_rainfall"]     = temperature * rainfall

    # CSI — using training set stats
    temp_norm     = (temperature - stats["temp_mean"])     / stats["temp_std"]
    humidity_norm = (humidity    - stats["humidity_mean"]) / stats["humidity_std"]
    rainfall_norm = (rainfall    - stats["rainfall_mean"]) / stats["rainfall_std"]
    feats["CSI"]  = (0.4 * abs(temp_norm) +
                     0.3 * abs(humidity_norm) +
                     0.3 * abs(rainfall_norm))

    # pH bands
    feats["ph_acidic"]   = 1 if ph < 6.0 else 0
    feats["ph_neutral"]  = 1 if 6.0 <= ph <= 7.5 else 0
    feats["ph_alkaline"] = 1 if ph > 7.5 else 0

    # Rainfall bands
    feats["rainfall_low"]    = 1 if rainfall < 60 else 0
    feats["rainfall_medium"] = 1 if 60 <= rainfall < 150 else 0
    feats["rainfall_high"]   = 1 if rainfall >= 150 else 0

    return pd.DataFrame([feats])[feature_names]

# ============================================================
# CROP EMOJI MAP
# ============================================================
crop_emoji = {
    "apple": "🍎", "banana": "🍌", "barley": "🌾", "blackgram": "🫘",
    "chickpea": "🫘", "coconut": "🥥", "coffee": "☕", "cotton": "🌿",
    "grapes": "🍇", "jute": "🌿", "kidneybeans": "🫘", "lentil": "🫘",
    "maize": "🌽", "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫘",
    "muskmelon": "🍈", "orange": "🍊", "papaya": "🍑", "pigeonpeas": "🫘",
    "pomegranate": "🍎", "rice": "🌾", "soybean": "🫘", "sugarcane": "🎋",
    "watermelon": "🍉", "wheat": "🌾"
}

# ============================================================
# UI — HEADER
# ============================================================
st.title("🌾 Crop Recommendation System")
st.markdown("*Climate-aware crop recommendation with Explainable AI*")
st.markdown("---")

# ============================================================
# UI — SIDEBAR INPUTS
# ============================================================
st.sidebar.header("🧪 Enter Soil & Climate Parameters")
st.sidebar.markdown("Adjust the sliders to match your field conditions.")

N           = st.sidebar.slider("Nitrogen (N) — kg/ha",          0,   140,  50)
P           = st.sidebar.slider("Phosphorus (P) — kg/ha",        5,   145,  50)
K           = st.sidebar.slider("Potassium (K) — kg/ha",         5,   205,  50)
temperature = st.sidebar.slider("Temperature — °C",               8.0, 44.0, 25.0, step=0.1)
humidity    = st.sidebar.slider("Humidity — %",                  14.0, 100.0, 65.0, step=0.1)
ph          = st.sidebar.slider("pH Value",                       3.5,  9.5,  6.5, step=0.1)
rainfall    = st.sidebar.slider("Rainfall — mm",                  20.0, 300.0, 100.0, step=0.5)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🌱 Get Recommendation", use_container_width=True)

# ============================================================
# UI — MAIN CONTENT
# ============================================================
if not predict_btn:
    # Landing state
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\n\nAdjust the soil and climate sliders on the left sidebar to match your field conditions.")
    with col2:
        st.info("**Step 2**\n\nClick **Get Recommendation** to run the AI model.")
    with col3:
        st.info("**Step 3**\n\nView your recommended crop, confidence score, and SHAP explanation.")

    st.markdown("---")
    st.markdown("### About This System")
    st.markdown("""
    This system uses a **Tuned Random Forest** model trained on 3,463 agronomic samples 
    across **26 crop classes**. It achieves **88.25% cross-validation accuracy** (95% CI: 87.25%–89.25%).

    **Explainability** is provided via SHAP (SHapley Additive exPlanations), showing exactly 
    which soil and climate factors drove each recommendation.

    | Parameter | Description |
    |---|---|
    | N, P, K | Soil macronutrients (kg/ha) |
    | Temperature | Mean ambient temperature (°C) |
    | Humidity | Relative humidity (%) |
    | pH | Soil acidity/alkalinity |
    | Rainfall | Annual rainfall (mm) |
    """)

else:
    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------
    input_df = engineer_features(N, P, K, temperature, humidity, ph, rainfall, stats)

    probabilities  = model.predict_proba(input_df)[0]
    top3_idx       = np.argsort(probabilities)[::-1][:3]
    top3_crops     = le.inverse_transform(top3_idx)
    top3_probs     = probabilities[top3_idx]

    predicted_crop = top3_crops[0]
    confidence     = top3_probs[0]
    emoji          = crop_emoji.get(predicted_crop, "🌱")

    # --------------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------------
    st.markdown("## 🎯 Recommendation Result")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.success(f"### {emoji} {predicted_crop.upper()}")
        st.markdown(f"**Confidence:** {confidence*100:.1f}%")
        st.progress(float(confidence))

    with col2:
        st.metric("Top Prediction",    f"{top3_crops[0].capitalize()}", f"{top3_probs[0]*100:.1f}%")
        st.metric("2nd Alternative",   f"{top3_crops[1].capitalize()}", f"{top3_probs[1]*100:.1f}%")

    with col3:
        st.metric("3rd Alternative",   f"{top3_crops[2].capitalize()}", f"{top3_probs[2]*100:.1f}%")
        csi_val = input_df["CSI"].values[0]
        stress  = "Low 🟢" if csi_val < 0.5 else ("Medium 🟡" if csi_val < 1.0 else "High 🔴")
        st.metric("Climate Stress Index", f"{csi_val:.3f}", stress)

    st.markdown("---")

    # --------------------------------------------------------
    # INPUT SUMMARY TABLE
    # --------------------------------------------------------
    st.markdown("### 📋 Your Input Parameters")
    input_summary = pd.DataFrame({
        "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
                      "Temperature", "Humidity", "pH", "Rainfall"],
        "Value":     [f"{N} kg/ha", f"{P} kg/ha", f"{K} kg/ha",
                      f"{temperature} °C", f"{humidity} %",
                      f"{ph}", f"{rainfall} mm"]
    })
    st.table(input_summary)

    st.markdown("---")

    # --------------------------------------------------------
    # SHAP EXPLANATION
    # --------------------------------------------------------
    st.markdown("### 🔍 Why This Crop Was Recommended (SHAP Explanation)")
    st.markdown("The chart below shows which factors most influenced this recommendation. "
                "🔴 Red bars pushed the prediction **toward** this crop. "
                "🔵 Blue bars pushed **away** from it.")

    with st.spinner("Generating SHAP explanation..."):
        try:
            explainer   = shap.TreeExplainer(model)
            explanation = explainer(input_df)
            # explanation.values shape: (1, 22, 26)

            predicted_class_idx = list(le.classes_).index(predicted_crop)

            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                explanation[0, :, predicted_class_idx],
                show=False
            )
            plt.title(f"SHAP Explanation — Recommended: {predicted_crop.capitalize()}", 
                      fontsize=13, pad=15)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP plot could not be generated: {e}")
            st.markdown("**Top contributing features (by model importance):**")
            importance_df = pd.DataFrame({
                "Feature":    feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False).head(8)
            st.bar_chart(importance_df.set_index("Feature"))

    st.markdown("---")

    # --------------------------------------------------------
    # TOP 3 ALTERNATIVES BAR CHART
    # --------------------------------------------------------
    st.markdown("### 📊 Top 3 Crop Probabilities")

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    colors = ["#2ecc71", "#3498db", "#95a5a6"]
    bars   = ax2.barh(
        [f"{crop_emoji.get(c,'🌱')} {c.capitalize()}" for c in top3_crops[::-1]],
        top3_probs[::-1] * 100,
        color=colors[::-1],
        edgecolor="white"
    )
    for bar, val in zip(bars, top3_probs[::-1] * 100):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=10)
    ax2.set_xlabel("Confidence (%)")
    ax2.set_xlim(0, 105)
    ax2.set_title("Model Confidence — Top 3 Crops")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.caption("Model: Tuned Random Forest | CV Accuracy: 88.25% ± 0.51% | "
               "Dataset: 3,463 samples, 26 crops | XAI: SHAP TreeExplainer")