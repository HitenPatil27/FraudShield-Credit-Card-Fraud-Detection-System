import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="FraudShield - Credit Card Fraud Detection",
    layout="wide"
)

# Configuration
MODEL_PATH = "model/dl_fraud_model.h5"      # Deep Learning Model
SCALER_PATH = "model/scaler.pkl"            # StandardScaler
DEFAULT_THRESHOLD = 0.5                     # ⬅ Default threshold

# These are the 30 feature columns used in training
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
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Could not load DL model from {MODEL_PATH}: {e}")
        raise

    try:
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Could not load scaler from {SCALER_PATH}: {e}")
        raise

    return model, scaler

model, scaler = load_artifacts()
threshold = DEFAULT_THRESHOLD   # ⬅ use default

# UI
st.title("🛡️ FraudShield – Credit Card Fraud Detection (Deep Learning Model)")

st.markdown(
    """
Upload a CSV file containing transactions to detect fraud.

🔹 Model: **Deep Neural Network (Keras)**  
🔹 Threshold for fraud prediction: **0.5**
🔹 CSV must contain 30 columns used in model training (`Time, V1…V28, Amount`)
"""
)

st.sidebar.header("Options")
show_preview = st.sidebar.checkbox("Show CSV preview", value=True)
show_confusion = st.sidebar.checkbox("Show Confusion Matrix if Class present", value=True)
download_name = st.sidebar.text_input("Download filename", value="fraud_predictions_dl.csv")

uploaded_file = st.file_uploader("Upload credit card transactions CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

    if show_preview:
        st.write("### 🔍 CSV Preview")
        st.dataframe(df.head())

    # Check columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing}")
        st.stop()

    X = df[FEATURE_COLS].values

    # Scale
    try:
        Xs = scaler.transform(X)
    except Exception as e:
        st.error(f"Scaling error: {e}")
        st.stop()

    # Predict from DL model
    scores = model.predict(Xs).ravel()               # probability of fraud
    preds = (scores >= threshold).astype(int)        # label based on threshold

    out_df = df.copy().reset_index(drop=True)
    out_df["Fraud_Probability"] = scores
    out_df["Predicted_Label"] = preds

    # Summary
    st.write("### Prediction Summary")
    total = len(out_df)
    fraud = int((preds == 1).sum())
    normal = total - fraud

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Predicted FRAUD", fraud)
    col3.metric("Predicted NORMAL", normal)

    st.write("### Sample Predictions")
    st.dataframe(out_df.head(20))

    # Confusion matrix (if Class exists)
    if "Class" in df.columns and show_confusion:
        st.write("### Confusion Matrix & Report")

        y_true = df["Class"].astype(int)
        y_pred = preds

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        cm_df = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Actual Normal", "Actual Fraud"],
            columns=["Pred Normal", "Pred Fraud"]
        )

        st.table(cm_df)
        st.text(classification_report(y_true, y_pred, digits=4))

    # Download
    csv_bytes = out_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download predictions CSV",
        data=csv_bytes,
        file_name=download_name,
        mime="text/csv"
    )

else:
    st.info("Upload a CSV file to begin prediction.")
