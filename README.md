# Fraud Detection in Online Transactions using Machine Learning

## 📌 Project Overview
This is my MCA Major Project at Amity University Online.  
The project focuses on detecting fraudulent online transactions using advanced Machine Learning techniques such as **SMOTE, Random Forest, XGBoost, and Isolation Forest**.

---

## 🎯 Objectives
- Build a scalable, real-time fraud detection pipeline.
- Handle **extreme class imbalance** using SMOTE & undersampling.
- Engineer behavioural features (velocity, geospatial anomalies, spending deviations).
- Compare supervised (Logistic Regression, Random Forest, XGBoost) and unsupervised (Isolation Forest, One-Class SVM) models.
- Evaluate using **Precision, Recall, F1-score, AUCPR** instead of plain accuracy.

---

## 📂 Dataset
- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- Transactions: 284,807  
- Fraud cases: 492 (0.17%)  

---

## ⚙️ Tech Stack
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, GitHub  

---

## 📊 Methodology
1. **Data Preprocessing** – Cleaning, scaling, balancing with SMOTE.  
2. **Feature Engineering** – Velocity profiles, geospatial anomalies, spending deviations.  
3. **Model Training** – Logistic Regression, Random Forest, XGBoost, Isolation Forest.  
4. **Evaluation** – Confusion Matrix, Precision-Recall Curve, AUCPR.  

---

## 📈 Results
- XGBoost + SMOTE gave the best balance of **high recall** and **low false positives**.  
- Isolation Forest detected **zero-day fraud patterns**.  
- AUCPR significantly higher than baseline Logistic Regression.  

---

## 🚀 How to Run
```bash
git clone https://github.com/riddhiitees-glitch/Fraud-Detection-Online-Transactions-ML.git
cd Fraud-Detection-Online-Transactions-ML
pip install -r requirements.txt
jupyter notebook notebooks/FraudDetection.ipynb
