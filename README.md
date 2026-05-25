# 🩺 Breast Cancer Detection — ML Classification Pipeline

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)]()

> End-to-end ML pipeline for binary breast cancer classification using the Wisconsin Diagnostic dataset. Achieves **~95% recall** and **92% precision** to minimise false negatives in early-detection clinical workflows — developed during AI/ML Internship at **InternPE**.

---

## 📌 Problem Statement

In clinical cancer screening, **false negatives (missed cancer) are far more dangerous than false positives**. This pipeline is optimised for maximum recall — it is better to flag a healthy patient for further testing than to miss a malignant case.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| **Recall (sensitivity)** | **~95%** ← primary objective |
| Precision | ~92% |
| F1-Score | ~93% |
| AUC-ROC | ~0.98 |
| False Negatives on test set | Minimised |

---

## ✨ Pipeline Stages

```
Raw Data (569 samples, 30 features)
    │
    ├── 1. EDA & Visualisation (class balance, correlations, distributions)
    │
    ├── 2. Preprocessing (StandardScaler, stratified train/test split)
    │
    ├── 3. Model Comparison via 5-Fold Stratified CV
    │       ├── Logistic Regression
    │       ├── Random Forest ✅ best recall
    │       ├── Gradient Boosting
    │       └── SVM (RBF kernel)
    │
    ├── 4. Hyperparameter Tuning (GridSearchCV — recall-optimised)
    │
    ├── 5. Final Evaluation (confusion matrix, AUC-ROC, classification report)
    │
    └── 6. Feature Importance Analysis (top predictive features)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Evaluation | AUC-ROC, Precision-Recall curves |

---

## 📁 Project Structure

```
Breast-Cancer-Detection/
│
├── breast_cancer_classifier.py   # Full ML pipeline
├── requirements.txt              # Dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

```bash
git clone https://github.com/AnmolPandey9119/Breast-Cancer-Detection.git
cd Breast-Cancer-Detection

pip install -r requirements.txt
python breast_cancer_classifier.py
```

The script uses sklearn's built-in Wisconsin dataset — no external data download needed.

---

## 🔬 Key Insights

- **Worst area**, **worst concave points**, and **mean concave points** are the top 3 most discriminative features
- Gradient Boosting and Random Forest consistently outperform Logistic Regression on recall
- Stratified splitting is critical given the class imbalance (~37% malignant, ~63% benign)
- Scaling (StandardScaler) significantly boosts SVM and Logistic Regression performance

---

## 🔮 Future Enhancements

- [ ] Deploy as REST API (FastAPI + Docker) for clinical integration
- [ ] Add SHAP explainability for each prediction
- [ ] Experiment with neural network approaches (PyTorch)
- [ ] Add support for custom patient data via CSV upload

---

## 👤 Author

**Anmol Pandey** — ML Engineer & AI Developer  
Built during **AI/ML Internship @ InternPE** (Aug–Sep 2025)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/anmol-pandey-240105376)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/AnmolPandey9119)

> ⭐ If this project was useful, please star the repo!
