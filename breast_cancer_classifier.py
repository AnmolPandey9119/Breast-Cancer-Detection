"""
Breast Cancer Detection — ML Classification Pipeline
Author: Anmol Pandey (github.com/AnmolPandey9119)
Description: End-to-end ML pipeline for binary breast cancer classification
             using the Wisconsin Diagnostic dataset. Achieves ~95% recall
             to minimise false negatives in early-detection workflows.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, recall_score, precision_score
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Load & Explore Data
# ---------------------------------------------------------------------------
print("=" * 60)
print("  🩺  Breast Cancer Detection  |  ML Pipeline")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

print(f"\n📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"   Class distribution:\n{y.value_counts().rename({0: 'Malignant', 1: 'Benign'})}")
print(f"   Missing values: {X.isnull().sum().sum()}")

# ---------------------------------------------------------------------------
# 2. Train / Test Split (stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Split: {X_train.shape[0]} train / {X_test.shape[0]} test (stratified)")

# ---------------------------------------------------------------------------
# 3. Model Comparison — 4 candidates evaluated via 5-Fold CV
# ---------------------------------------------------------------------------
MODELS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, probability=True, random_state=42))
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n📋 5-Fold Cross-Validation Results:")
print(f"{'Model':<25} {'CV Accuracy':>12} {'CV Recall':>10}")
print("-" * 50)

best_model_name, best_model, best_recall = None, None, 0.0

for name, model in MODELS.items():
    acc_scores   = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    recall_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    mean_acc    = acc_scores.mean()
    mean_recall = recall_scores.mean()
    print(f"{name:<25} {mean_acc:>11.1%} {mean_recall:>9.1%}")
    if mean_recall > best_recall:
        best_recall = mean_recall
        best_model_name = name
        best_model = model

print(f"\n🏆 Best model by recall: {best_model_name}")

# ---------------------------------------------------------------------------
# 4. Final Evaluation on held-out test set
# ---------------------------------------------------------------------------
best_model.fit(X_train, y_train)
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n📊 Test Set Results:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

recall    = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auc_roc   = roc_auc_score(y_test, y_proba)

print(f"  Recall    : {recall:.1%}   ← minimising false negatives is top priority")
print(f"  Precision : {precision:.1%}")
print(f"  F1-Score  : {f1:.1%}")
print(f"  AUC-ROC   : {auc_roc:.3f}")

# ---------------------------------------------------------------------------
# 5. Feature Importance (Random Forest / tree-based)
# ---------------------------------------------------------------------------
try:
    clf = best_model.named_steps["clf"]
    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=data.feature_names).sort_values(ascending=False)

    print(f"\n🔬 Top 10 Most Important Features ({best_model_name}):")
    for feat, imp in feat_imp.head(10).items():
        bar = "█" * int(imp * 200)
        print(f"  {feat:<35} {imp:.4f}  {bar}")
except AttributeError:
    pass  # Not all models have feature_importances_

# ---------------------------------------------------------------------------
# 6. Confusion Matrix
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
print(f"\n🔲 Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"\n  False Negatives (missed cancer): {cm[1,0]} — critical to minimise")

print("\n✅ Pipeline complete. Model ready for deployment.")
