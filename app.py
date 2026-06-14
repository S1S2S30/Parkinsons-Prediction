import streamlit as st
import pandas as pd
import joblib

# =====================================
# CONFIG
# =====================================
st.set_page_config(page_title="Parkinson's Detection App", page_icon="🩺", layout="wide")

# =====================================
# LOAD SAVED OBJECTS
# =====================================
pipeline = joblib.load("pipeline.pkl")
feature_means = joblib.load("feature_means.pkl")
all_features = joblib.load("all_features.pkl")
top_features = joblib.load("top_features.pkl")
reference_table = joblib.load("reference_table.pkl")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Introduction", "EDA", "Model & Prediction", "Reference Ranges", "Conclusion"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed by Sara Saeed | MPhil Data Science, PUCIT")

# =====================================
# INTRODUCTION
# =====================================
if page == "Introduction":
    st.title("🩺 Parkinson's Disease Detection")
    st.markdown("""
    This application predicts **Parkinson's Disease** using
    **voice measurement features**.

    **Techniques used:**
    - Feature Engineering
    - Feature Selection (Top 10 Features)
    - Logistic Regression Pipeline
    - Class imbalance handling via class weights

    👉 Go to **Model & Prediction** to test the app.
    👉 Go to **Reference Ranges** to see typical Healthy vs Parkinson's values.
    """)

# =====================================
# EDA (STATIC INFO)
# =====================================
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.info("EDA was performed in Jupyter Notebook.")
    st.markdown("""
    - Class imbalance checked
    - Feature distributions analyzed
    - Correlation & statistical tests performed
    - Top 10 features selected for the prediction UI
    """)

    st.subheader("ROC Curve of Best Model")
    try:
        st.image("outputs_no_smote/roc_curve_best_model.png")
    except Exception:
        st.warning("ROC curve image not found.")

    st.subheader("Selected Features (by coefficient)")
    try:
        st.image("outputs_no_smote/selected_features_abscoef.png")
    except Exception:
        st.warning("Feature importance image not found.")

# =====================================
# MODEL & PREDICTION
# =====================================
elif page == "Model & Prediction":

    st.title("🩺 Parkinson's Prediction")
    st.markdown("### Enter voice measurement values")

    # -------- INPUT FORM --------
    with st.form("prediction_form"):
        input_data = {}

        cols = st.columns(2)
        for i, feature in enumerate(top_features):
            default_val = float(feature_means.get(feature, 0))
            with cols[i % 2]:
                input_data[feature] = st.number_input(
                    feature,
                    value=default_val,
                    key=feature
                )

        submit = st.form_submit_button("🔍 Predict")

    # -------- PREDICTION --------
    if submit:
        # create full feature row (same as training)
        input_df = pd.DataFrame([feature_means])
        for k, v in input_data.items():
            input_df[k] = v

        input_df = input_df[all_features]  # SAME order as training

        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠ **Parkinson's Detected**  \nProbability: **{prob:.2%}**")
        else:
            st.success(f"✅ **Healthy**  \nProbability: **{(1 - prob):.2%}**")

        st.caption("💡 Tip: Check the **Reference Ranges** tab to see how your values compare to typical Healthy vs Parkinson's averages.")

# =====================================
# REFERENCE RANGES
# =====================================
elif page == "Reference Ranges":
    st.title("📋 Reference Ranges — Healthy vs Parkinson's")
    st.markdown("""
    This table shows the **average values** of each feature
    for Healthy individuals vs Parkinson's patients in the training dataset.

    ⚠️ **Note:** Parkinson's is predicted using a *combination* of all 10 features together —
    no single feature alone determines the result. Use this table only as a general reference.
    """)

    display_table = reference_table.rename(columns={
        "feature": "Feature",
        "healthy_avg": "Healthy (Average)",
        "parkinsons_avg": "Parkinson's (Average)"
    })

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    st.markdown("""
    **General Patterns Observed:**
    - Lower `MDVP:Fo(Hz)` and `MDVP:Flo(Hz)` (pitch) tend to appear more in Parkinson's cases
    - Higher `MDVP:Shimmer`, `MDVP:APQ`, `spread1/spread2`, and `PPE_y` are associated with Parkinson's
    - Lower `HNR_y` (Harmonic-to-Noise Ratio) and `DFA_x` are associated with Parkinson's
    """)

# =====================================
# CONCLUSION
# =====================================
elif page == "Conclusion":
    st.title("📌 Conclusion")
    st.markdown("""
    - Model predicts Parkinson's Disease using **voice features**
    - Only the **top 10 most important features** are shown for simplicity
    - Class imbalance handled via **class weights** (no SMOTE)
    - App is built with a single end-to-end **scikit-learn pipeline**

    ✅ Ready for demo / portfolio
    """)