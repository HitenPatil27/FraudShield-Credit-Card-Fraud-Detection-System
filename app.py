# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title="FraudShield - Credit Card Fraud Detection",
    layout="wide"
)

# Configuration
MODEL_PATH = "model/best_fraud_model.pkl"
SCALER_PATH = "model/scaler.pkl"
THRESHOLD_PATH = "model/threshold_info.json"  # or cost_optimal_threshold.json

# These are the 30 feature columns used in training (order matters)
FEATURE_COLS = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

# Load artifacts
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Could not load model from {MODEL_PATH}: {e}")
        raise

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"❌ Could not load scaler from {SCALER_PATH}: {e}")
        raise

    try:
        with open(THRESHOLD_PATH, "r") as f:
            info = json.load(f)
        threshold = float(info.get("cost_optimal_threshold", info.get("best_f1_threshold", 0.5)))
    except Exception:
        st.warning("threshold_info.json not found or invalid. Using default threshold = 0.5")
        threshold = 0.5

    return model, scaler, threshold

model, scaler, threshold = load_artifacts()

# UI Layout
st.title("🛡️ FraudShield – Credit Card Fraud Detection (CSV Upload)")
st.markdown(
    """
Upload a CSV file of transactions and the system will:

1. Predict **fraud score** and **fraud/normal label** for each transaction  
2. Show how many are predicted **Fraud** and how many **Normal**  
3. If the CSV contains the true label column `Class`, it will also show a **confusion matrix** and **classification report**  

**Note:**  
The CSV must contain at least these 30 columns (used to train the model):  
`Time, V1..V28, Amount`
"""
)

st.sidebar.header("Options")
show_preview = st.sidebar.checkbox("Show CSV preview", value=True)
show_confusion = st.sidebar.checkbox("Show confusion metrics (if Class present)", value=True)
download_name = st.sidebar.text_input("Download predictions filename", value="fraud_predictions.csv")

# File Upload
uploaded_file = st.file_uploader("Upload credit card transactions CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if show_preview:
        st.markdown("### Preview of uploaded data")
        st.dataframe(df.head())

    # Check required feature columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    # Extract features in correct order
    X = df[FEATURE_COLS].values

    # Scale
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        st.error(f"Error while scaling features: {e}")
        st.stop()

    # Get fraud scores
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(Xs)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(Xs)
    else:
        scores = model.predict(Xs)

    # Labels by threshold
    preds = (scores >= threshold).astype(int)

    # Attach to dataframe
    out_df = df.copy().reset_index(drop=True)
    out_df["y_score"] = scores
    out_df["y_pred_threshold"] = preds

    # 1) Show overall prediction counts
    st.markdown("### Prediction Summary")

    total = len(out_df)
    fraud_pred = int((out_df["y_pred_threshold"] == 1).sum())
    normal_pred = total - fraud_pred

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total)
    with col2:
        st.metric("Predicted FRAUD (1)", fraud_pred)
    with col3:
        st.metric("Predicted NORMAL (0)", normal_pred)

    st.markdown("### Sample of predictions")
    st.dataframe(out_df.head(20))

    # 2) Show confusion matrix & report if true labels available
    if "Class" in out_df.columns and show_confusion:
        st.markdown("### Confusion Matrix and Metrics (using true Class)")

        y_true = out_df["Class"].astype(int)
        y_pred = out_df["y_pred_threshold"].astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual 0 (Normal)", "Actual 1 (Fraud)"],
            columns=["Pred 0 (Normal)", "Pred 1 (Fraud)"]
        )

        st.markdown("**Confusion Matrix:**")
        st.table(cm_df)

        st.markdown("**Classification Report:**")
        st.text(classification_report(y_true, y_pred, digits=4))

    # 3) Download predictions
    st.markdown("### Download predictions with scores and labels")
    csv_bytes = out_df.to_csv(index=False).encode()
    st.download_button(
        label="Download predictions CSV",
        data=csv_bytes,
        file_name=download_name,
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to start prediction.")
