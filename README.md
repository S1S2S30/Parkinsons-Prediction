# 🩺 Parkinson's Disease Detection

A Machine Learning web app that predicts the likelihood of **Parkinson's Disease** based on voice measurement features, using a Logistic Regression pipeline.

---

## 📌 Project Overview

Parkinson's Disease affects speech patterns in measurable ways — changes in pitch, jitter, shimmer, and harmonic ratios. This project builds a classification model that analyzes these voice biomarkers to predict whether a person is likely to have Parkinson's Disease.

The app includes an interactive Streamlit interface with multiple sections: introduction, exploratory data analysis, live prediction, reference ranges, and conclusion.

---

## 📊 Dataset

| Detail | Value |
|---|---|
| Total Samples | 195 |
| Original Features | 47 |
| Target Variable | `status` (0 = Healthy, 1 = Parkinson's) |
| Class Distribution | 147 Parkinson's, 48 Healthy |

The dataset contains biomedical voice measurements such as fundamental frequency (Fo, Fhi, Flo), jitter, shimmer, harmonic-to-noise ratio (HNR), and nonlinear dynamic features (DFA, PPE, spread1, spread2).

---

## 🧹 Data Preprocessing & Feature Engineering

- Removed identifier columns (`subject#`, `measurement_id`, `name`, `sex`) and the target column from features
- Engineered new features: pitch range, pitch ratio, jitter mean/std, shimmer mean/std
- Missing value imputation using **mean strategy**
- Feature scaling using **StandardScaler**
- Feature selection using **SelectKBest (f_classif)** — top 10 features selected

### Top 10 Selected Features
```
age, DFA_x, MDVP:Fo(Hz), MDVP:Flo(Hz), MDVP:Shimmer,
MDVP:APQ, HNR_y, spread1, spread2, PPE_y
```

---

## 🤖 Model & Results

| Detail | Value |
|---|---|
| Algorithm | Logistic Regression (`class_weight='balanced'`) |
| Train/Test Split | 80/20 (stratified) |
| **Accuracy** | **87.18%** |
| **ROC-AUC** | **95.17%** |
| Imbalance Handling | Class weights (no SMOTE) |

The full preprocessing + model is bundled into a single **scikit-learn Pipeline** for consistent inference.

---

## 🖥️ App Features

- **Introduction** — overview of the project and techniques used
- **EDA** — ROC curve and feature importance visualizations
- **Model & Prediction** — interactive form to input voice features and get instant prediction with probability
- **Reference Ranges** — comparison table of Healthy vs Parkinson's average values for each feature
- **Conclusion** — summary of approach and results

---

## 🛠️ Tech Stack

- **Python**
- **Pandas / NumPy** — Data handling
- **Scikit-learn** — Preprocessing, feature selection, model & pipeline
- **Streamlit** — Web app interface
- **Matplotlib** — Visualizations (ROC curve, feature importance)

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/S1S2S30/Parkinsons-Prediction.git
cd Parkinsons-Prediction
pip install -r requirments.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
Parkinsons-Prediction/
├── app.py                          # Streamlit web app
├── trainging-new-code.ipynb        # Model training notebook
├── eda-and-statistical-test.ipynb  # EDA & statistical analysis
├── pipeline.pkl                    # Trained preprocessing + model pipeline
├── feature_means.pkl               # Mean values for all features
├── all_features.pkl                # List of all features used in training
├── top_features.pkl                # Top 10 selected features (shown in UI)
├── reference_table.pkl             # Healthy vs Parkinson's averages
├── outputs_no_smote/                # Saved plots (ROC curve, feature importance)
├── requirments.txt
└── README.md
```

---

## ⚠️ Disclaimer

This tool is built for **educational and research purposes only** and is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider for medical concerns.

---

## 🔮 Future Improvements

- Try ensemble models (Random Forest, SVM, Voting/Stacking Classifiers)
- Add SHAP for model explainability
- Expand dataset size for better generalization
- Deploy live demo on HuggingFace Spaces

---

## 👩‍💻 Author

**Sara Saeed** — MPhil Data Science, PUCIT, Lahore
