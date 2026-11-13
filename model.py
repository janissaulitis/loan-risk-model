"""Loan risk model: which new loans are likely to go 30+ days late.

Three approaches, all trained on months 1-18 and tested on months 19-24 (the "future" they never saw):
  * Rule      - what a team might do without a model: flag anyone with a past late payment or payment-to-income above 45%.
  * Explainable model (logistic regression) - every prediction can be broken down into reasons a person can read.
  * Benchmark (gradient boosting) - a more powerful "black box", used only to check how much accuracy we give up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).parent / "data" / "loans.csv"
TRAIN_UNTIL = 18  # months 1-18 train, 19-24 test

CATEGORICAL = ["market", "channel", "product"]
NUMERIC = ["payment_to_income", "pti_over_40", "employment_months", "new_job_heavy_payment", "previous_loans",
           "previous_late", "clean_repeat_customer", "back_within_30_days", "night_application", "amount_eur"]

# Plain-language names for the reasons shown to a person
LABEL = {
    "payment_to_income": "Monthly payment vs income", "pti_over_40": "Payment above 40% of income",
    "employment_months": "Time in current job", "new_job_heavy_payment": "New job and a heavy payment",
    "previous_loans": "Number of earlier loans", "previous_late": "Earlier late payments",
    "clean_repeat_customer": "Repaid earlier loans on time", "back_within_30_days": "Back for a new loan within 30 days",
    "night_application": "Applied late at night", "amount_eur": "Loan amount",
    "market": "Market", "channel": "Channel", "product": "Product",
}


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Turns raw application fields into the signals the model uses. The same function runs for training and for new
    applications, so the two can never drift apart."""
    x = df.copy()
    x["pti_over_40"] = (x.payment_to_income > 0.40).astype(int)
    x["new_job_heavy_payment"] = ((x.employment_months < 6) & (x.payment_to_income > 0.30)).astype(int)
    x["clean_repeat_customer"] = np.minimum(x.previous_loans, 5) * (x.previous_late == 0)
    x["back_within_30_days"] = ((x.days_since_last_loan >= 0) & (x.days_since_last_loan < 30)).astype(int)
    x["night_application"] = ((x.application_hour < 6) | (x.application_hour >= 23)).astype(int)
    x["employment_months"] = x.employment_months.clip(upper=60)
    x["previous_late"] = x.previous_late.clip(upper=3)
    return x[CATEGORICAL + NUMERIC]


def rule_score(df: pd.DataFrame) -> np.ndarray:
    return ((df.previous_late > 0) | (df.payment_to_income > 0.45)).astype(float).to_numpy()


@dataclass
class Trained:
    explainable: Pipeline
    benchmark: Pipeline
    train: pd.DataFrame
    test: pd.DataFrame


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


def train(df: pd.DataFrame | None = None) -> Trained:
    df = load() if df is None else df
    tr, te = df[df.month <= TRAIN_UNTIL], df[df.month > TRAIN_UNTIL]
    prep = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL),
                              ("num", StandardScaler(), NUMERIC)])
    explainable = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=1.0))])
    explainable.fit(features(tr), tr.late_30)
    benchmark = Pipeline([("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                                                     remainder="passthrough")),
                          ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=3, min_samples_leaf=80,
                                                                  random_state=0))])
    benchmark.fit(features(tr), tr.late_30)
    return Trained(explainable, benchmark, tr, te)


def risk(m: Trained, df: pd.DataFrame) -> np.ndarray:
    return m.explainable.predict_proba(features(df))[:, 1]


def reasons(m: Trained, row: pd.DataFrame, top: int = 3) -> list[tuple[str, float]]:
    """How much each signal pushes this one application's risk up (+) or down (-), compared with an average applicant.
    For a logistic regression this breakdown is exact, not an approximation."""
    prep, lr = m.explainable.named_steps["prep"], m.explainable.named_steps["model"]
    x = prep.transform(features(row))
    x = x.toarray() if hasattr(x, "toarray") else x
    base = prep.transform(features(m.train))
    base = np.asarray(base.mean(axis=0)).ravel()
    contrib = (x[0] - base) * lr.coef_[0]
    names = prep.get_feature_names_out()
    by_signal: dict[str, float] = {}
    for n, c in zip(names, contrib):
        key = n.split("__", 1)[1]
        key = next((cat for cat in CATEGORICAL if key.startswith(cat + "_")), key)
        by_signal[key] = by_signal.get(key, 0.0) + c
    ranked = sorted(by_signal.items(), key=lambda kv: -abs(kv[1]))[:top]
    return [(LABEL.get(k, k), v) for k, v in ranked]


def evaluate(m: Trained) -> dict:
    te, y = m.test, m.test.late_30.to_numpy()
    scores = {"Rule": rule_score(te), "Explainable model": risk(m, te),
              "Benchmark (black box)": m.benchmark.predict_proba(features(te))[:, 1]}
    out = {}
    for name, s in scores.items():
        k = max(1, int(round(0.10 * len(te))))
        top = np.argsort(-s, kind="stable")[:k]
        out[name] = {
            "auc": roc_auc_score(y, s),
            "precision_top10": y[top].mean(),        # of the 10% flagged, how many really went late
            "capture_top10": y[top].sum() / y.sum(),  # of all late loans, how many were in the flagged 10%
        }
    return {"metrics": out, "n_test": len(te), "late_rate_test": y.mean(), "late_rate_train": m.train.late_30.mean()}


def calibration(m: Trained, bins: int = 8) -> pd.DataFrame:
    p, y = risk(m, m.test), m.test.late_30.to_numpy()
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    d = pd.DataFrame({"predicted": p, "actual": y, "bin": q}).groupby("bin").mean()
    return d


def by_market(m: Trained, flag_share: float = 0.10) -> pd.DataFrame:
    te = m.test.copy()
    te["risk"] = risk(m, te)
    cut = np.quantile(te.risk, 1 - flag_share)
    te["flagged"] = te.risk >= cut
    rows = []
    for mk, g in te.groupby("market"):
        late = g[g.late_30 == 1]
        rows.append({"Market": mk, "Loans": len(g), "Late rate": g.late_30.mean(), "Flagged": g.flagged.mean(),
                     "Late loans caught": late.flagged.mean() if len(late) else np.nan,
                     "AUC": roc_auc_score(g.late_30, g.risk) if g.late_30.nunique() > 1 else np.nan})
    return pd.DataFrame(rows)


def business_case(m: Trained, flag_share: float, call_cost: float, prevent_rate: float, loss_per_late: float,
                  monthly_loans: int) -> dict:
    """Monthly value of calling the riskiest customers early, scaled to a lender's monthly volume."""
    te = m.test
    p, y = risk(m, te), te.late_30.to_numpy()
    k = max(1, int(round(flag_share * len(te))))
    top = np.argsort(-p, kind="stable")[:k]
    scale = monthly_loans / len(te)
    calls = k * scale
    caught = y[top].sum() * scale
    prevented = caught * prevent_rate
    saved = prevented * loss_per_late
    cost = calls * call_cost
    return {"calls": calls, "late_among_called": caught, "prevented": prevented, "saved": saved, "cost": cost,
            "net": saved - cost}
