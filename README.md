# Which loans will go late? A loan risk model with reasons

Demo for a fictional consumer lender. Predicts which new loans are likely to go 30+ days late, so the collections team can contact those customers early, and shows the reasons behind every score.

- `data/make_loans.py` – generates about 6,500 invented loans over 24 months in 6 markets (no gender, age, nationality or similar)
- `model.py` – a simple rule, an explainable logistic regression and a gradient boosting benchmark, trained on months 1–18 and tested on months 19–24; per-loan reasons, calibration, results by market, business case
- `app.py` – Streamlit app: call list, try an application, evaluation, business case, fairness & limits

Result on the unseen test months: the explainable model flags 10% of loans, and 40% of those really go late, against 21% for the simple rule and a 15% average. The black-box benchmark is no better (AUC 0.712 vs 0.717), so the explainable model is kept.
