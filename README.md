# 🏦 End-to-End Banking Customer Churn Analytics & Predictive Modeling

<p align="left">

</p>

<p align="left">

<a href="https://www.python.org/">
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
</a>

<a href="https://streamlit.io/">
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
</a>

<a href="https://pandas.pydata.org/">
<img src="https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas&logoColor=white" alt="Pandas">
</a>

<a href="https://plotly.com/python/">
<img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?style=flat-square&logo=plotly&logoColor=white" alt="Plotly">
</a>

<a href="https://scikit-learn.org/">
<img src="https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
</a>

<a href="https://xgboost.readthedocs.io/">
<img src="https://img.shields.io/badge/XGBoost-Predictive_Modeling-189AB4?style=flat-square" alt="XGBoost">
</a>

<a href="https://www.microsoft.com/en-us/sql-server">
<img src="https://img.shields.io/badge/SQL_Server-Analytics-CC2927?style=flat-square&logo=microsoftsqlserver&logoColor=white" alt="SQL Server">
</a>

</p>

---

## 📌 Overview

**End-to-End Banking Customer Churn Analytics & Predictive Modeling** is an end-to-end data analytics and machine-learning project focused on understanding customer churn and identifying customers who may be at risk of leaving a banking service.

The project combines **SQL, Python, statistical analysis, machine learning, and an interactive Streamlit dashboard** into a single retention-intelligence workflow.

The objective is not only to predict churn, but also to translate historical customer behavior into actionable business insights.

---


## 🚀 Live Application

<p align="left">

<a href="https://end-to-end-banking-customer-churn-analytics-predictive-modeling.streamlit.app/">
<img src="https://img.shields.io/badge/🚀_Open_Live_Dashboard-FF4B4B?style=for-the-badge" alt="Open Live Dashboard">
</a>

</p>

---

Explore the interactive dashboard to analyze customer churn, compare machine-learning models, and evaluate customer risk.
---

## 🎯 Business Objective

The project addresses the following business questions:

- How many customers are churning?
- What is the overall churn rate?
- Which geographic markets have the highest number of churned customers?
- How does churn vary by customer age?
- Does the number of products relate to churn behavior?
- How does active status affect churn?
- Which machine-learning model performs best?
- Which customers should be prioritized for retention?

---

## 📊 Project Snapshot

| Metric | Value |
|---|---:|
| Total Customers | 10,000 |
| Churned Customers | 2,037 |
| Overall Churn Rate | 20.37% |
| Average Balance | $119,827 |
| Average Salary | $100,090 |
| ML Models Evaluated | 5 |

---

## 🧠 Analytics Workflow

```text
Raw Banking Data
       ↓
Data Cleaning & Validation
       ↓
SQL Data Analysis
       ↓
Exploratory Data Analysis
       ↓
Statistical Testing
       ↓
Feature Engineering & Preprocessing
       ↓
Machine Learning
       ↓
Model Evaluation & Comparison
       ↓
Churn Probability
       ↓
Customer Risk Identification
       ↓
Retention Decision Support
```
---

## 🔬 Machine Learning Models

Five classification models were evaluated:

1. Logistic Regression
2. Random Forest
3. SVM Polynomial
4. SVM RBF
5. XGBoost

### Model Comparison

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.652 | 0.328 | 0.437 | 0.743 |
| Random Forest | 0.660 | 0.482 | 0.558 | 0.831 |
| SVM Polynomial | 0.671 | 0.496 | **0.570** | 0.817 |
| SVM RBF | **0.711** | 0.475 | 0.569 | 0.830 |
| XGBoost | 0.430 | **0.749** | 0.547 | **0.839** |

### Key Model Findings

- **SVM RBF** achieved the highest Recall at **0.711**.
- **XGBoost** achieved the highest Precision at **0.749**.
- **SVM Polynomial** achieved the highest F1-score at **0.570**.
- **XGBoost** achieved the highest ROC-AUC at **0.839**.

The preferred model therefore depends on the business objective rather than selecting a model based on only one evaluation metric.

---

## 📈 Business Analytics

The Streamlit dashboard provides interactive analysis of:

- Customer and churn KPIs
- Churn by Geography
- Churn by Age
- Churn by Number of Products
- Churn by Active Status
- Customer-level churn prediction
- Model performance comparison

### Key Business Findings

- The overall historical churn rate is **20.37%**.
- The **UK** has the highest number of churned customers in the dataset.
- Customers with **4 products** show the highest observed churn rate at **100%**.
- Inactive customers have a higher churn rate than active customers in this dataset.
- Churn rate varies considerably across customer age groups.

> These findings describe patterns observed in the dataset and should not automatically be interpreted as causal relationships.

---

## 🗂️ Project Structure

```text
BankChurnProject/
│
├── data/
│   ├── processed/
│   │   ├── account.csv
│   │   ├── demographic.csv
│   │   └── location.csv
│   └── raw/
│       └── raw_data.xlsx
│
├── eda_queries/
│   ├── ChurnRate Accross Genders.sql
│   ├── Difference Average ChrunRate.sql
│   └── Dynamic Parameters.sql
│
├── models/
│   └── churn_model_bundle.pkl
│
├── notebook/
│   ├── statistical_testing/
│   │   └── statistical_testing.ipynb
│   └── test.sql
│
├── predictive_modelling/
│   ├── experiments/
│   │   ├── exp_LogisticReg.ipynb
│   │   ├── exp_RandomForest.ipynb
│   │   ├── exp_SVM_POL.ipynb
│   │   ├── exp_SVM_RBF.ipynb
│   │   ├── exp_XGBoost.ipynb
│   │   └── model_comparison.ipynb
│   │
│   ├── processed_data/
│   │   ├── dataset_bundle.pkl
│   │   └── preprocessing_bundle.pkl
│   │
│   ├── evaluation_script.py
│   └── preprocessing.py
│
├── scripts/
│   ├── data cleaning/
│   ├── data_ingestion/
│   └── utils/
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Skills Demonstrated

- Python
- Pandas
- NumPy
- SQL
- SQL Server
- Exploratory Data Analysis
- Data Cleaning
- Statistical Hypothesis Testing
- Statistical Inference
- Feature Engineering
- Machine Learning
- Classification
- Model Evaluation
- Churn Prediction
- Streamlit
- Interactive Data Visualization
- Business Analytics
- Git & GitHub

---

## 📦 Technologies

**Data & Analysis:**  
Python, Pandas, NumPy, SQL, SQL Server

**Statistics:**  
SciPy, Hypothesis Testing, Statistical Inference

**Machine Learning:**  
Scikit-Learn, XGBoost

**Visualization:**  
Plotly

**Application:**  
Streamlit

**Development:**  
Jupyter Notebook, Git, GitHub


## ⚠️ Disclaimer

This project is intended for portfolio purposes.

The analysis demonstrates a data-driven approach to customer churn analysis, prediction, and retention decision support. Model predictions should be validated against real-world banking data, business requirements, and appropriate governance processes before being used in production.

---
