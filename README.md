# CreditSense — AI Loan Approval Risk Predictor

An end-to-end machine learning web application that predicts whether a loan
application is likely to be **Approved** or **Rejected**, along with a
probability score and flagged risk factors.

Built to demonstrate a complete ML + full-stack pipeline:
**Python (data + ML) → Flask (API) → React (UI)**

---

## Project structure

```
credit-sense/
├── ml/
│   ├── generate_data.py      # builds the synthetic loan-applicant dataset
│   ├── train_model.py        # EDA, preprocessing, model training & evaluation
│   ├── loan_data.csv         # generated dataset (1500 rows)
│   ├── model.joblib          # trained Logistic Regression model
│   ├── scaler.joblib         # StandardScaler for numeric features
│   ├── encoders.joblib       # LabelEncoders for categorical features
│   ├── feature_meta.joblib   # column order/metadata used by the API
│   ├── metrics_report.json   # accuracy / precision / recall / F1 / ROC-AUC
│   └── *.png                 # EDA + evaluation charts
├── backend/
│   ├── app.py                # Flask REST API serving the model
│   └── requirements.txt
└── frontend/
    └── index.html            # React UI (CDN-based, no build step needed)
```

## How to run it

### 1. Train the model (optional — already-trained artifacts are included)
```bash
cd ml
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python generate_data.py
python train_model.py
```

### 2. Start the API
```bash
cd backend
pip install -r requirements.txt
python app.py
```
This starts the API at `http://localhost:5000`.

### 3. Open the frontend
Just open `frontend/index.html` directly in your browser (double-click it,
or use a simple static server like `npx serve frontend`). It talks to the
Flask API at `localhost:5000`.

---

## Model summary

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 77.7% | 73.7% |
| Precision | 73.5% | 69.3% |
| Recall | 63.7% | 54.0% |
| F1-score | 68.2% | 60.7% |
| ROC-AUC | 0.834 | 0.809 |

**Logistic Regression was selected** — it generalized slightly better and
is more interpretable (clear coefficients) for a financial use case where
explainability matters.

Top predictive features: **Debt-to-Income Ratio, Education, Credit Score,
Income, Existing Loans.**

## Deploying it live (for free)

See `DEPLOYMENT.md` for full step-by-step instructions to deploy the
backend on Render and the frontend on Netlify, both on free tiers.

## Note on the dataset

This project uses a **synthetically generated dataset** (`generate_data.py`)
modeled on real-world loan-application attributes, since no licensed
banking dataset was available. The generation logic encodes realistic
relationships (e.g. higher credit score and lower debt-to-income ratio
increase approval likelihood) plus random noise, so the ML pipeline behaves
like it would on real data. For a production version, swap in a real
dataset (e.g. a public Kaggle loan-prediction dataset) — the rest of the
pipeline (preprocessing → training → API → UI) stays the same.
