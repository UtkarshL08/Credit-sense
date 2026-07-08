import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


class PredictApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_predict_returns_prediction(self):
        payload = {
            "Age": 30,
            "Income": 60000,
            "LoanAmount": 150000,
            "LoanTerm": "60",
            "CreditScore": 700,
            "EmploymentYears": 4,
            "ExistingLoans": 1,
            "Dependents": "1",
            "Education": "Graduate",
            "SelfEmployed": "No",
            "PropertyArea": "Urban",
            "MaritalStatus": "Married",
            "DebtToIncomeRatio": 0.25,
        }

        response = self.client.post("/api/predict", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("approval_probability", data)
        self.assertIn("risk_factors", data)


if __name__ == "__main__":
    unittest.main()
