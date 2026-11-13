"""Generates data/loans.csv: about 6,500 invented consumer loans over 24 months, with whether each one went 30+ days late.

The data is synthetic, but built the way real lending data behaves: affordability (monthly payment vs income) drives
most of the risk, payment history matters a lot, stable employment helps, partner-channel and night-time applications
carry more risk, and the effect is not perfectly linear. There is noise, so no model can be perfect, and risk drifts up
slightly in the later months, so a model trained on the past is tested on a slightly different future.

Deliberately NOT included: gender, age, nationality, family status or anything else that could act as a protected
characteristic. Market (country of the loan) is included because pricing and rules differ by market.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 7000
MARKETS = {"LV": (1250, 0.00), "LT": (1300, -0.05), "PL": (1150, 0.05), "ES": (1500, 0.10), "RO": (950, 0.15), "PE": (700, 0.25)}

market = rng.choice(list(MARKETS), N, p=[0.14, 0.12, 0.24, 0.20, 0.18, 0.12])
income = np.array([rng.lognormal(np.log(MARKETS[m][0]), 0.38) for m in market]).round(-1).clip(350, 6000)
channel = rng.choice(["app", "web", "partner"], N, p=[0.55, 0.30, 0.15])
product = rng.choice(["instalment", "short"], N, p=[0.7, 0.3])
term = np.where(product == "short", 1, rng.choice([6, 12, 18, 24, 36], N, p=[0.15, 0.35, 0.2, 0.2, 0.1]))
amount = np.where(product == "short", rng.choice([100, 200, 300, 400, 500], N), rng.choice(np.arange(500, 5001, 250), N)).astype(float)
apr = np.where(product == "short", 0.0, rng.uniform(0.12, 0.35, N))
with np.errstate(divide="ignore", invalid="ignore"):
    monthly_payment = np.where(term == 1, amount * 1.08,
                               amount * (apr / 12) / (1 - (1 + apr / 12) ** -term.clip(min=2)))
pti = (monthly_payment / income).round(3)
employment_months = rng.gamma(2.0, 22, N).round().clip(0, 240)
previous_loans = rng.poisson(1.3, N)
previous_late = np.where(previous_loans > 0, rng.binomial(previous_loans, 0.18), 0)
days_since_last = np.where(previous_loans > 0, rng.integers(5, 720, N), -1)
hour = rng.choice(np.arange(24), N, p=np.r_[np.full(6, 0.01), np.full(18, (1 - 0.06) / 18)])
month = np.sort(rng.integers(1, 25, N))

z = (-2.75
     + 3.1 * np.minimum(pti, 0.8)
     + 0.7 * (pti > 0.40)                                   # past a point, affordability breaks down quickly
     + 0.55 * np.minimum(previous_late, 3)
     - 0.22 * np.minimum(previous_loans, 5) * (previous_late == 0)  # good repeat customers
     - 0.012 * np.minimum(employment_months, 60)
     + 0.50 * (channel == "partner") - 0.10 * (channel == "app")
     + 0.30 * (product == "short")
     + 0.35 * ((hour < 6) | (hour >= 23))
     + 0.45 * ((employment_months < 6) & (pti > 0.30))        # new job and a heavy payment
     + 0.40 * ((days_since_last >= 0) & (days_since_last < 30))  # back for another loan very soon
     + np.array([MARKETS[m][1] for m in market])
     + 0.15 * (month > 18)                                    # risk drifts up in the latest months
     + rng.normal(0, 0.55, N))
late = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)

df = pd.DataFrame({
    "loan_id": [f"L{100000 + i}" for i in range(N)], "month": month, "market": market, "channel": channel,
    "product": product, "amount_eur": amount, "term_months": term, "monthly_income_eur": income,
    "payment_to_income": pti, "employment_months": employment_months.astype(int), "previous_loans": previous_loans,
    "previous_late": previous_late, "days_since_last_loan": days_since_last, "application_hour": hour,
    "late_30": late,
})
# Only approved loans exist in a lender's history: applications above 60% payment-to-income are refused.
df = df[df.payment_to_income <= 0.60].reset_index(drop=True)
df["loan_id"] = [f"L{100000 + i}" for i in range(len(df))]
out = Path(__file__).with_name("loans.csv")
df.to_csv(out, index=False)
print(f"{len(df)} loans, {df.late_30.mean():.1%} went 30+ days late -> {out}")
print(df.groupby("month").late_30.mean().round(3).to_dict())
