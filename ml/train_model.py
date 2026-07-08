"""
train_model.py
----------------
Full ML pipeline for the Loan Approval Risk Predictor.

Steps:
1. Load data
2. Exploratory Data Analysis (EDA) + plots
3. Clean missing values
4. Encode categoricals, scale numerics
5. Train/test split
6. Train + compare two models: Logistic Regression vs Random Forest
7. Evaluate (accuracy, precision, recall, F1, ROC-AUC, confusion matrix)
8. Save the best model + preprocessing objects with joblib so the
   Flask API can load them later without retraining
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

OUT = Path(__file__).resolve().parent
sns.set_style("whitegrid")

# ----------------------------------------------------------------------
# 1. Load
# ----------------------------------------------------------------------
df = pd.read_csv(OUT / "loan_data.csv")

# ----------------------------------------------------------------------
# 2. EDA plots (saved as PNGs, also used later in the PDF report)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sns.countplot(x="Loan_Approved", data=df, ax=axes[0], palette=["#e74c3c", "#2ecc71"])
axes[0].set_title("Class Balance: Loan Approved vs Rejected")
axes[0].set_xticklabels(["Rejected (0)", "Approved (1)"])

sns.histplot(data=df, x="CreditScore", hue="Loan_Approved", kde=True, ax=axes[1],
             palette=["#e74c3c", "#2ecc71"], element="step")
axes[1].set_title("Credit Score Distribution by Outcome")
plt.tight_layout()
plt.savefig(OUT / "eda_overview.png", dpi=150)
plt.close()

plt.figure(figsize=(7, 5.5))
numeric_cols = ["Age", "Income", "LoanAmount", "LoanTerm", "CreditScore",
                 "EmploymentYears", "ExistingLoans", "Dependents",
                 "DebtToIncomeRatio", "Loan_Approved"]
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. Clean missing values
# ----------------------------------------------------------------------
df["EmploymentYears"] = df["EmploymentYears"].fillna(df["EmploymentYears"].median())
df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
df["SelfEmployed"] = df["SelfEmployed"].fillna(df["SelfEmployed"].mode()[0])

# ----------------------------------------------------------------------
# 4. Encode categoricals, scale numerics
# ----------------------------------------------------------------------
categorical_cols = ["Education", "SelfEmployed", "PropertyArea", "MaritalStatus"]
numeric_cols_model = ["Age", "Income", "LoanAmount", "LoanTerm", "CreditScore",
                       "EmploymentYears", "ExistingLoans", "Dependents", "DebtToIncomeRatio"]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[numeric_cols_model + categorical_cols]
y = df["Loan_Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols_model] = scaler.fit_transform(X_train[numeric_cols_model])
X_test_scaled[numeric_cols_model] = scaler.transform(X_test[numeric_cols_model])

# ----------------------------------------------------------------------
# 5 & 6. Train + compare models
# ----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    results[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    trained_models[name] = (model, preds, probs)

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model, best_preds, best_probs = trained_models[best_name]

print(json.dumps(results, indent=2))
print("Best model:", best_name)

# ----------------------------------------------------------------------
# 7. Detailed evaluation plots for the best model
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title(f"Confusion Matrix - {best_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, best_probs)
plt.figure(figsize=(5, 4.2))
plt.plot(fpr, tpr, label=f"ROC AUC = {results[best_name]['roc_auc']:.3f}", color="#2ecc71", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "roc_curve.png", dpi=150)
plt.close()

if best_name == "Random Forest":
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    importance_label = "Importance"
else:
    # For Logistic Regression, the absolute standardized coefficient size tells you
    # which features move the prediction most (a common, legitimate interpretability technique).
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=False)
    importance_label = "|Coefficient| (standardized)"
importances.index.name = "Feature"

plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title(f"Feature Importance - {best_name}")
plt.xlabel(importance_label)
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()
feature_importance_dict = importances.round(4).to_dict()

# ----------------------------------------------------------------------
# 8. Save model + preprocessing artifacts
# ----------------------------------------------------------------------
joblib.dump(best_model, OUT / "model.joblib")
joblib.dump(scaler, OUT / "scaler.joblib")
joblib.dump(encoders, OUT / "encoders.joblib")
joblib.dump({"numeric_cols": numeric_cols_model, "categorical_cols": categorical_cols,
             "feature_order": list(X.columns)}, OUT / "feature_meta.joblib")

report = {
    "best_model": best_name,
    "results": results,
    "feature_importance": feature_importance_dict,
    "n_train": len(X_train),
    "n_test": len(X_test),
    "classification_report": classification_report(y_test, best_preds, output_dict=True),
}
with open(OUT / "metrics_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nSaved model + scaler + encoders + metrics report.")
