# FraudShield — Credit Card Fraud Detection System

**FraudShield** is an AI-powered system designed to detect fraudulent credit card transactions using Machine Learning. The project handles class imbalance using **SMOTE**, applies robust preprocessing with **StandardScaler**, and trains models such as **Logistic Regression**, **Random Forest**, and **XGBoost** to identify suspicious activities in real time and prevent financial losses.

---

## Project Overview
This project focuses on detecting fraudulent transactions within highly imbalanced credit card data.  
It uses supervised learning and anomaly detection techniques to classify transactions as **fraudulent** or **legitimate**.  
The final trained model achieves high recall while maintaining precision to ensure minimal false alerts.

---

## Milestone 2 — Submission
This repository contains the deliverables for **Capstone Project Milestone 2 (Due: 29th October)**:

- ✅ **Training Dataset:** `creditcard.csv` (Kaggle — ULB Dataset)
- ✅ **Model Training Pipeline:** Available in the `notebook/` folder  
- ✅ **Trained Model:** Saved in the `model/` folder  

**Repository Structure:**

```
FraudShield-Credit-Card-Fraud-Detection-System/
├─ notebook/
│ └─ CreditCard_Fraud_Detection_Training.ipynb
├─ model/
│ └─ best_fraud_model.pkl
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

## How to Run the Project

### Option 1 — Google Colab (Recommended)
1. Open the notebook: `notebook/FraudShield_training.ipynb`
2. Upload `creditcard.csv` (or mount from Google Drive)
3. Run all cells sequentially  
4. Model and metrics will be generated and saved automatically

### Option 2 — Local Setup
```bash
git clone https://github.com/<your-username>/FraudShield-Credit-Card-Fraud-Detection-System.git
cd FraudShield-Credit-Card-Fraud-Detection-System
pip install -r requirements.txt
jupyter notebook notebook/FraudShield_training.ipynb
```

---

## Models & Techniques Used

- **Data Preprocessing:** StandardScaler
- **Balancing:** SMOTE (Synthetic Minority Oversampling Technique)
- **Models:** Logistic Regression, Random Forest, XGBoost
- **Evaluation Metrics:** Precision, Recall, F1-score, ROC-AUC
- **Model Saving:** .pkl format using joblib

## Results

| Model             | F1-Score | ROC-AUC |
|-------------------|----------|---------|
| Logistic Regression | 0.947864    | 0.949202   |
| Random Forest     | 0.999877    | 0.999877   |
| XGBoost           | 0.999429    | 0.999428   |


---

## Authors
- Hiten Patil  
- Smit Shedge  
- Vedant Patel  

B.Tech — Computer Science & Engineering  
Capstone Project (Deep Learning Subject)  
Instructor: Chintan Shah

---

## 🪪 License
This repository is created for academic and educational purposes.  
You may reuse the code with credit to the authors.
