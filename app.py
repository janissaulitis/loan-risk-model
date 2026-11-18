"""Loan risk model with reasons. Run: streamlit run app.py"""
import importlib

import numpy as np
import pandas as pd
import streamlit as st

import kit
import model as M

M = importlib.reload(M)  # Streamlit Cloud keeps old copies of imported files after an update; always use the latest
kit = importlib.reload(kit)

kit.page("Early Call", "Find the loans that will go late, and call before they do", mark="phone", icon="📞",
         meta=f"Trained on months 1–{M.TRAIN_UNTIL} · tested on months {M.TRAIN_UNTIL + 1}–24")

GREEN, RED = "#2fb985", "#e5484d"


@st.cache_resource(show_spinner="Training the model on 18 months of loans…")
def trained():
    return M.train()


m = trained()
ev = M.evaluate(m)
X = ev["metrics"]


def reason_text(rs):
    return " · ".join(f"{'▲' if v > 0 else '▼'} {name}" for name, v in rs)


def reason_chart(rs):
    kit.contrib([(n, v) for n, v in rs])


# ---------------------------------------------------------------- intro
hit, rule, base = X['Explainable model']['precision_top10'], X['Rule']['precision_top10'], ev['late_rate_test']
kit.answer("The call list works", f"{hit:.0%} of the loans on its call list really go 30+ days late",
           f"That is <b>{hit / rule:.1f}×</b> better than a simple rule ({rule:.0%}) and <b>{hit / base:.1f}×</b> better than "
           "calling at random. Tested on six months of loans it never saw, with the reasons behind every score.",
           [("Call-list hit rate", f"{hit:.0%}"), ("Simple rule", f"{rule:.0%}"), ("At random", f"{base:.0%}"),
            ("Accuracy (AUC)", f"{X['Explainable model']['auc']:.2f}")])
with st.container(horizontal=True):
    with st.popover("How it works"):
        st.markdown(
            "1. **Call list:** this month's loans, riskiest first, each with its top 3 reasons.\n"
            "2. **Try an application:** change income, payment, job time or history and watch the risk and reasons move.\n"
            "3. **How good is it:** compared with a simple rule and with a more complex `black box` model, on months the model never saw.\n"
            "4. **Business case:** what calling the riskiest 10% early is worth per month, with your own assumptions.\n"
            "5. **Fairness & limits:** what the model deliberately does not use, how it behaves by market, and what to watch."
        )
st.caption(f"{len(m.train) + len(m.test):,} invented loans over 24 months in 6 markets.")

tab_list, tab_try, tab_eval, tab_biz, tab_fair = st.tabs(
    ["Call list", "Try an application", "How good is it", "Business case", "Fairness & limits"])

# ---------------------------------------------------------------- call list
with tab_list:
    month = m.test[m.test.month == m.test.month.max()].copy()
    month["risk"] = M.risk(m, month)
    month = month.sort_values("risk", ascending=False)
    k = max(1, int(round(0.10 * len(month))))
    st.markdown(f"**Month {int(month.month.iloc[0])}: {len(month)} new loans. The {k} riskiest go on the call list.**")
    top = month.head(k)
    rows = []
    for _, r in top.iterrows():
        rs = M.reasons(m, top.loc[[r.name]])
        rows.append({"Loan": r.loan_id, "Market": r.market, "Amount €": int(r.amount_eur),
                     "Risk": r.risk, "Top reasons": reason_text(rs),
                     "Actually went late": "yes" if r.late_30 else "no"})
    kit.table(rows, [kit.col("Loan", "Loan", "muted"), kit.col("Market", "Market", "lang"),
                     kit.col("Amount €", "Amount €", "num", fmt=lambda v: f"{v:,}"),
                     kit.col("Risk", "Risk", "meter", {"warn": 0.3, "bad": 0.5}), kit.col("Top reasons", "Top reasons", "wrap"),
                     kit.col("Went late", "Actually went late", "chip", {"yes": "bad", "no": "good"})], max_height=440)
    hit = top.late_30.mean()
    st.caption(f"▲ raises risk · ▼ lowers risk. Hindsight check: {hit:.0%} of this list really went late, "
               f"against {month.late_30.mean():.0%} of all loans that month.")
    pick = st.selectbox("See the full breakdown for", top.loan_id.tolist())
    one = top[top.loan_id == pick]
    st.markdown(f"**{pick}: {float(one.risk.iloc[0]):.0%} risk of going 30+ days late**")
    reason_chart(M.reasons(m, one, top=6))

# ---------------------------------------------------------------- try an application
with tab_try:
    st.markdown("Change the application and watch the risk and the reasons move.")
    result = st.container()
    a, b, c3 = st.columns(3)
    income = a.number_input("Monthly income €", 300, 8000, 1200, 50)
    payment = a.number_input("Monthly loan payment €", 10, 4000, 180, 10)
    jobm = b.slider("Months in current job", 0, 120, 24)
    prev = b.slider("Earlier loans", 0, 8, 1)
    prev_late = b.slider("…of which paid late", 0, prev, 0) if prev else 0
    market = c3.selectbox("Market", ["LV", "LT", "PL", "ES", "RO", "PE"])
    channel = c3.selectbox("Channel", ["app", "web", "partner"])
    product = c3.selectbox("Product", ["instalment", "short"])
    night = c3.toggle("Applied between 23:00 and 06:00")
    soon = c3.toggle("Back for a new loan within 30 days", disabled=prev == 0)
    app_row = pd.DataFrame([{
        "market": market, "channel": channel, "product": product, "amount_eur": payment * 10,
        "payment_to_income": payment / income, "employment_months": jobm, "previous_loans": prev,
        "previous_late": prev_late, "days_since_last_loan": (10 if soon else 200) if prev else -1,
        "application_hour": 2 if night else 14}])
    p = float(M.risk(m, app_row)[0])
    level = "high" if p >= 0.30 else "medium" if p >= 0.15 else "low"
    with result:
        l, r = st.columns([2, 3], gap="large")
        with l:
            kit.answer("Risk of going 30+ days late", f"{p:.0%}, {level}",
                       f"Payment is <b>{payment / income:.0%} of income</b>. The average loan is {ev['late_rate_test']:.0%}.",
                       tone="bad" if p >= 0.30 else "warn" if p >= 0.15 else "good")
        with r:
            st.markdown("#### Why")
            reason_chart(M.reasons(m, app_row, top=5))
        st.divider()

# ---------------------------------------------------------------- evaluation
with tab_eval:
    st.markdown(f"Tested on **{ev['n_test']:,} loans from months {M.TRAIN_UNTIL + 1}–24**, which the model never saw "
                "during training. That's how it would meet the future.")
    kit.table([{"a": k, "auc": f"{v['auc']:.3f}", "p": v["precision_top10"], "c": v["capture_top10"],
                "u": "In use" if k == "Explainable model" else ""} for k, v in X.items()],
              [kit.col("Approach", "a", "strong"), kit.col("", "u", "chip", {"In use": "good"}), kit.col("Accuracy (AUC)", "auc", "num"),
               kit.col("Late among the flagged 10%", "p", "meter"), kit.col("Share of all late loans caught", "c", "meter")])
    st.markdown(
        f"- The model nearly **doubles the hit rate** of a simple rule: {X['Explainable model']['precision_top10']:.0%} "
        f"of the people it flags really go late, against {X['Rule']['precision_top10']:.0%} for the rule.\n"
        f"- A more complex model (gradient boosting) is **no better** "
        f"({X['Benchmark (black box)']['auc']:.3f} vs {X['Explainable model']['auc']:.3f}), so the explainable one is kept: "
        "every score can be explained to a customer, an auditor or a regulator.\n"
        "- No model gets close to perfect: much of what makes a loan go late (a sudden job loss, illness) "
        "isn't in any application. That is normal for credit risk.")
    st.markdown("**Does the predicted risk match reality?**")
    cal = M.calibration(m)
    kit.pairs([(f"Group {i + 1}", float(r.predicted), float(r.actual)) for i, r in enumerate(cal.itertuples())],
              ("Predicted", "Actually late"), fmt=lambda x: f"{x:.1%}")
    st.caption(f"Loans grouped from lowest to highest predicted risk. The ranking holds, but late payments rose from "
               f"{ev['late_rate_train']:.1%} in the training months to {ev['late_rate_test']:.1%} in the test months, "
               "so the model slightly under-predicts. In production it would be re-checked monthly and re-calibrated "
               "when that gap grows.")

# ---------------------------------------------------------------- business case
with tab_biz:
    st.markdown("**What is calling the riskiest customers early worth?** Set your own assumptions.")
    a, b = st.columns(2)
    monthly = a.number_input("New loans per month", 100, 100_000, 2000, 100)
    share = a.slider("Share of loans to call", 0.02, 0.30, 0.10, 0.01, format="%.2f")
    cost = a.number_input("Cost of one call €", 0.5, 30.0, 4.0, 0.5)
    prevent = b.slider("Late payments an early call prevents", 0.05, 0.60, 0.25, 0.05, format="%.2f",
                       help="E.g. 0.25 = one in four late payments avoided by a reminder or a payment plan.")
    loss = b.number_input("Cost of one late loan €", 20, 2000, 150, 10,
                          help="Collection work, fees written off, part of the loan lost.")
    bc = M.business_case(m, share, cost, prevent, loss, monthly)
    kit.answer("With your numbers", f"Calling the riskiest {share:.0%} early is worth about €{bc['net']:,.0f} a month",
               f"€{bc['saved']:,.0f} of late-payment costs avoided, minus €{bc['cost']:,.0f} spent on calls.",
               [("Calls a month", f"{bc['calls']:,.0f}"), ("Late loans among them", f"{bc['late_among_called']:,.0f}"),
                ("Late payments prevented", f"{bc['prevented']:,.0f}")], tone="good" if bc["net"] > 0 else "warn")
    st.caption("Uses the model's real hit rate on the test months; the money assumptions are yours. "
               "Calling at random would reach far fewer late loans for the same cost.")

# ---------------------------------------------------------------- fairness
with tab_fair:
    st.markdown("**What the model deliberately does not use:** gender, age, nationality, family status, or anything "
                "that stands in for them. Market (the country of the loan) is used, because prices and rules differ by country.")
    st.markdown("**How it behaves by market** (flagging the riskiest 10% overall):")
    bm = M.by_market(m)
    kit.table(bm.to_dict("records"),
              [kit.col("Market", "Market", "lang"), kit.col("Loans", "Loans", "num", fmt=lambda v: f"{v:,}"),
               kit.col("Late rate", "Late rate", "meter", {"warn": 0.15, "bad": 0.25}), kit.col("Flagged", "Flagged", "meter"),
               kit.col("Late loans caught", "Late loans caught", "meter"), kit.col("AUC", "AUC", "num", fmt=lambda v: f"{v:.2f}")])
    st.caption("Markets with more late loans get more calls, which is the point. What to watch: the model ranks "
               "equally well in every market (similar AUC), so no market is served by a worse model.")
    st.markdown(
        "**Limits, said plainly**\n"
        "- The data is invented. A real version starts with the lender's own history, and the same tests.\n"
        "- It's meant for **who to contact early**, not for approving or refusing loans. A model that decides on "
        "credit for people counts as high-risk under the EU AI Act and needs much more: documentation, human "
        "oversight and the right to an explanation.\n"
        "- Risk drifts over time. Check every month: hit rate of the call list, predicted vs actual late rate, "
        "results by market. Re-train when they slip.")

kit.footer("scikit-learn · model.py")
