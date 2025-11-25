# FraudShield — Credit Card Fraud Detection System

**FraudShield** is an AI-powered system designed to detect fraudulent credit card transactions using both **Machine Learning** and **Deep Learning (DNN)**.  

The project handles class imbalance using **SMOTE**, applies robust preprocessing with **StandardScaler**, and trains models such as **Logistic Regression**, **Random Forest**, **XGBoost**, and a **Deep Neural Network (DNN)** to identify suspicious activities in real time and prevent financial losses.

---

## Project Overview

This project focuses on detecting fraudulent transactions within highly imbalanced credit card data.  
It uses supervised learning and anomaly detection techniques to classify transactions as **fraudulent** or **legitimate**.  

The final trained models achieve **high recall** while maintaining **good precision**, ensuring minimal false alerts and better fraud coverage.  
On top of classical ML models, we also trained a **Deep Neural Network (DNN)** model that achieved around **98% accuracy** on the test set.

A user-friendly **Streamlit application** is provided for real-time fraud detection, allowing users to:

- Upload transaction data in CSV format  
- Process it through the trained model  
- View predictions and download results  

The app also visualizes results with summary statistics, confusion matrices, and classification reports, making it ideal for quick assessments in banking or fintech environments.

---

## Repository Structure
```
FraudShield-Credit-Card-Fraud-Detection-System/
├─ notebook/
│ ├─ CreditCard_Fraud_Detection_Training.ipynb
| └─ CreditCard_Fraud_Detection_Training_with_DL.ipynb
├─ model/
│ ├─ best_fraud_model.pkl
│ ├─ scaler.pkl
│ ├─threshold_info.json
│ └─ dnn_fraud_model.h5
├─ app.py
└─ README.md
```

---

## Dataset Details
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Samples:** 284,807
- **Fraud Cases:** 492 (~0.17%)
- **Features:** 30 anonymized PCA components + `Time`, `Amount`
- **Label:** `Class` (0 = Legitimate, 1 = Fraud)

---

## Application Use
FraudShield provides an interactive web application built with Streamlit for seamless fraud detection. Users can upload transaction data in CSV format, process it through the trained model, and receive instant predictions. The app visualizes results with summary statistics, confusion matrices, and classification reports, making it ideal for quick assessments in banking or fintech environments.

---

## Application Preview
🟢 Below, insert the Streamlit application screenshots:
<img width="1655" height="945" alt="1" src="https://github.com/user-attachments/assets/5683c063-98bb-4263-a371-b3ab0b502b15" />
Dashboard

<img width="1661" height="945" alt="2" src="https://github.com/user-attachments/assets/4211abe3-30f4-4399-92f7-56d0c223ddab" />
<img width="1659" height="939" alt="3" src="https://github.com/user-attachments/assets/3183278c-8b53-49ee-9f97-5038e7f43893" />
Uploading CSV

<img width="1652" height="903" alt="4" src="https://github.com/user-attachments/assets/a70269e4-5345-4442-ae46-4c90da1db82f" />
Output
---

## How to Run the Project

### Run the Streamlit Application
🔹 Using Streamlit
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

🔹 Upload CSV Format  
Your CSV must include exactly these columns:

| Column | Description          |
|--------|----------------------|
| Time   | Transaction time     |
| V1     | PCA component 1      |
| V2     | PCA component 2      |
| ...    | ... (up to V28)      |
| Amount | Transaction amount   |

Class column is optional — if present, the app will show evaluation metrics.

---

## Models & Techniques Used
- **Data Preprocessing:** StandardScaler
- **Balancing:** SMOTE (Synthetic Minority Oversampling Technique)
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Evaluation Metrics:** Precision, Recall, F1-score, ROC-AUC
- **Model Saving:** .pkl format using joblib

## Results
| Model                | F1-Score  | ROC-AUC   |
|----------------------|-----------|-----------|
| Logistic Regression  | 0.947864  | 0.949202  |
| Random Forest        | 0.999877  | 0.999877  |
| XGBoost              | 0.999429  | 0.999428  |

## Deep Learning Model (DNN)
Model: Deep Neural Network (fully connected)
Test Accuracy: ≈ 98%
The DNN captures deeper non-linear patterns in the transaction data and provides competitive performance alongside tree-based models like Random Forest and XGBoost.

---

## Real-World Applications
| Industry       | Use Case                          |
|----------------|-----------------------------------|
| Banks          | Detect fraudulent card swipes     |
| FinTech Apps   | Secure online transactions        |
| E-commerce     | Prevent fake purchases & chargebacks |
| Payment Gateways | Monitor abnormal spending activity |

