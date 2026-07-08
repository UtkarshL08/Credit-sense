"""
generate_data.py
-----------------
Creates a synthetic but realistic "Loan Approval" dataset.

Why synthetic data?
In a real interview/production setting you would use a public dataset
(e.g. Kaggle's "Loan Prediction Practice Problem" or a bank's anonymised
records). Here we GENERATE data with the same kind of features and the
same kind of real-world relationships (higher credit score + lower debt
ratio => higher approval chance) so the full pipeline (EDA -> training ->
evaluation -> API -> UI) can be built and demonstrated end-to-end.
This is a completely standard and honest practice for portfolio projects
when you don't have access to a licensed dataset.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
np.random.seed(42)
N = 1500

# ---- Feature generation -------------------------------------------------
age = np.random.randint(21, 60, N)
income = np.random.normal(55000, 22000, N).clip(15000, 200000).round(0)
loan_amount = np.random.normal(150000, 80000, N).clip(20000, 600000).round(0)
loan_term = np.random.choice([12, 24, 36, 60, 84, 120, 180, 240, 360], N)
credit_score = np.random.normal(650, 90, N).clip(300, 900).round(0)
employment_years = np.random.exponential(5, N).clip(0, 35).round(1)
existing_loans = np.random.poisson(1.1, N).clip(0, 6)
dependents = np.random.choice([0, 1, 2, 3, 4], N, p=[0.35, 0.25, 0.2, 0.13, 0.07])
education = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.72, 0.28])
self_employed = np.random.choice(["Yes", "No"], N, p=[0.18, 0.82])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.38, 0.36, 0.26])
marital_status = np.random.choice(["Married", "Single"], N, p=[0.65, 0.35])

debt_to_income = (loan_amount / loan_term) / (income / 12)
debt_to_income = debt_to_income.clip(0, 3)

df = pd.DataFrame({
    "Age": age,
    "Income": income,
    "LoanAmount": loan_amount,
    "LoanTerm": loan_term,
    "CreditScore": credit_score,
    "EmploymentYears": employment_years,
    "ExistingLoans": existing_loans,
    "Dependents": dependents,
    "Education": education,
    "SelfEmployed": self_employed,
    "PropertyArea": property_area,
    "MaritalStatus": marital_status,
    "DebtToIncomeRatio": debt_to_income.round(3),
})

# ---- Target generation (the "ground truth" rule + noise) ----------------
# Build a latent "approval score" from sensible real-world weights, then
# turn it into a probability and sample a binary outcome from it. This
# keeps the relationship learnable but not deterministic (like real data).
score = (
    0.0075 * (df["CreditScore"] - 650)
    + 0.000022 * (df["Income"] - 55000)
    - 2.1 * df["DebtToIncomeRatio"]
    + 0.08 * df["EmploymentYears"]
    - 0.30 * df["ExistingLoans"]
    - 0.15 * df["Dependents"]
    + np.where(df["Education"] == "Graduate", 0.55, -0.25)
    + np.where(df["SelfEmployed"] == "Yes", -0.40, 0.15)
    + np.where(df["PropertyArea"] == "Urban", 0.25, np.where(df["PropertyArea"] == "Semiurban", 0.40, -0.20))
)

prob_approved = 1 / (1 + np.exp(-score))
noise = np.random.normal(0, 0.05, N)  # real-world unpredictability
prob_approved = (prob_approved + noise).clip(0.02, 0.98)
df["Loan_Approved"] = np.random.binomial(1, prob_approved)

# A few missing values, like real-world data (good talking point: how you handled them)
for col in ["EmploymentYears", "SelfEmployed", "Dependents"]:
    mask = np.random.rand(N) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv(BASE_DIR / "loan_data.csv", index=False)
print("Saved dataset:", df.shape)
print(df["Loan_Approved"].value_counts(normalize=True))
