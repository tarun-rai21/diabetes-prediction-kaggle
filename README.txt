# 🩺 Diabetes Prediction – Kaggle Playground Series

## 📌 Competition
Diabetes Prediction Challenge from the Kaggle Tabular Playground Series.

## 🎯 Objective
Predict the probability that a patient is diagnosed with diabetes using demographic and health-related features.

---

## 📊 Dataset
Synthetic healthcare dataset with features such as:

- Age
- BMI
- Gender
- Education Level
- Smoking Status
- Income Level

Target variable: **diagnosed_diabetes**

> Dataset not included due to Kaggle competition rules.

---

## ⚙️ ML Pipeline

This project follows a modular machine learning structure similar to real-world ML systems.

### Steps:
1. Data preprocessing (missing values, datatype handling)
2. Feature engineering (categorical encoding)
3. Model training using XGBoost
4. Train-validation split for performance evaluation
5. Prediction on competition test dataset
6. Submission file generation

---

## 🧪 Experiments

| Experiment | Description |
|-----------|------------|
| Baseline Model | Initial pipeline setup |
| Improved Model | Refined preprocessing and training configuration |

---

## 🏗 Project Structure

diabetes-prediction-kaggle/
│
├── notebooks/
│   ├── 01_baseline_model.ipynb       # Baseline experiment pipeline
│   └── 02_improved_model.ipynb       # Refined experiment with improved setup
│
├── src/
│   ├── preprocess.py                 # Data cleaning and preprocessing logic
│   ├── features.py                   # Feature preparation and encoding
│   ├── model.py                      # XGBoost training and validation
│   └── predict.py                    # Submission file generation
│
├── submissions/
│   ├── submission_v1.csv             # Baseline model predictions
│   └── submission_v2.csv             # Improved model predictions
│
├── requirements.txt                  # Project dependencies
├── .gitignore                        # Files ignored by Git
└── README.md                         # Project documentation

---

## 🧠 Skills Demonstrated

- End-to-end ML workflow
- Feature engineering
- Model validation
- Modular pipeline design
- Kaggle competition workflow

---

## 🚀 Technologies Used

- Python
- Pandas
- Scikit-learn
- XGBoost
