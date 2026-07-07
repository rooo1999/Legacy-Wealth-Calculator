"""
Legacy Wealth Calculator
=========================
A robust retirement + legacy planning tool for wealth management clients.

Run with:  streamlit run legacy_wealth_calculator.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Dict

st.set_page_config(page_title="Legacy Wealth Calculator", layout="wide", page_icon="🏛️")

# ──────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Child:
    name: str
    current_age: int
    education_cost_today: float        # cost in today's rupees, lump sum at age 18
    marriage_cost_today: float          # cost in today's rupees, lump sum at age 25
    education_age: int = 18
    marriage_age: int = 25


@dataclass
class Inputs:
    current_age: int
    retirement_age: int
    life_expectancy: int
    net_worth_liquid: float
    net_worth_illiquid: float           # e.g. primary residence, not used for retirement income
    monthly_savings: float
    monthly_expenses: float

    pre_retirement_return: float
    post_retirement_return: float
    income_growth_rate: float
    general_inflation: float
    education_inflation: float
    car_inflation: float
    healthcare_inflation: float

    owns_house: bool
    house_target_cost_today: float
    house_purchase_age: int

    car_cost_today: float
    car_upgrade_every_n_years: int

    children: List[Child] = field(default_factory=list)

    existing_emi_monthly: float = 0.0
    emi_end_age: int = 0

    parents_support_monthly: float = 0.0
    parents_support_end_age: int = 0

    capital_gains_tax_rate: float = 0.12   # effective drag on withdrawals


# ──────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ──────────────────────────────────────────────────────────────────────────

def goal_outflow_for_year(age: int, inp: Inputs) -> Dict[str, float]:
    """Returns a breakdown of goal-driven outflows due in the year the client turns `age`."""
    outflows = {"education": 0.0, "marriage": 0.0, "car": 0.0, "house": 0.0}
    years_from_now = age - inp.current_age

    # Education & marriage costs, inflated at education_inflation from today to the due date
    for child in inp.children:
        child_age_then = child.current_age + years_from_now
        if child_age_then == child.education_age:
            years_to_inflate = child.education_age - child.current_age
            years_to_inflate = max(years_to_inflate, 0)
            outflows["education"] += child.education_cost_today * (1 + inp.education_inflation) ** years_to_inflate
        if child_age_then == child.marriage_age:
            years_to_inflate = child.marriage_age - child.current_age
            years_to_inflate = max(years_to_inflate, 0)
            outflows["marriage"] += child.marriage_cost_today * (1 + inp.general_inflation) ** years_to_inflate

    # Car upgrades, inflated at car_inflation, recurring every N years from today
    if inp.car_upgrade_every_n_years > 0 and years_from_now > 0:
        if years_from_now % inp.car_upgrade_every_n_years == 0:
            outflows["car"] += inp.car_cost_today * (1 + inp.car_inflation) ** years_from_now

    # House purchase, one-time
    if (not inp.owns_house) and age == inp.house_purchase_age:
        years_to_inflate = inp.house_purchase_age - inp.current_age
        outflows["house"] += inp.house_target_cost_today * (1 + inp.general_inflation) ** max(years_to_inflate, 0)

    return outflows


def run_deterministic_projection(inp: Inputs):
    """
    Year-by-year simulation across the client's full lifespan.
    Phase 1 (accumulation): current_age -> retirement_age, uses pre_retirement_return
    Phase 2 (decumulation): retirement_age -> life_expectancy, uses post_retirement_return
    """
    rows = []

    net_worth = inp.net_worth_liquid
    annual_savings = inp.monthly_savings * 12
    annual_expense = inp.monthly_expenses * 12

    shortfall_age = None

    for age in range(inp.current_age, inp.life_expectancy + 1):
        is_retired = age >= inp.retirement_age
        ret_rate = inp.post_retirement_return if is_retired else inp.pre_retirement_return

        goals = goal_outflow_for_year(age, inp)
        goal_total = sum(goals.values())

        emi = inp.existing_emi_monthly * 12 if age < inp.emi_end_age else 0.0
        parent_support = inp.parents_support_monthly * 12 if age < inp.parents_support_end_age else 0.0

        # healthcare cost grows faster than general expenses, rises with age post-60
        healthcare_loading = 0.0
        if age >= 60:
            years_post_60 = age - 60
            healthcare_loading = (annual_expense * 0.08) * (1 + inp.healthcare_inflation) ** years_post_60

        if not is_retired:
            inflow = annual_savings
            outflow = goal_total + emi + parent_support + healthcare_loading
            net_worth = net_worth * (1 + ret_rate) + inflow - outflow
            annual_savings *= (1 + inp.income_growth_rate)
            annual_expense *= (1 + inp.general_inflation)
        else:
            # withdrawal grossed up for capital gains tax drag
            withdrawal_needed = annual_expense + healthcare_loading + goal_total + parent_support + emi
            withdrawal_grossed_up = withdrawal_needed / (1 - inp.capital_gains_tax_rate)

            if net_worth <= 0:
                # Corpus already depleted: don't let it compound into a runaway
                # negative number. Freeze at zero and record the unmet need
                # for that year instead, so the shortfall is visible without
                # producing a meaningless figure.
                net_worth = 0.0
                if shortfall_age is None:
                    shortfall_age = age
            else:
                net_worth = net_worth * (1 + ret_rate) - withdrawal_grossed_up
                if net_worth < 0:
                    net_worth = 0.0
                    if shortfall_age is None:
                        shortfall_age = age

            annual_expense *= (1 + inp.general_inflation)

        rows.append({
            "age": age,
            "phase": "Retirement" if is_retired else "Accumulation",
            "net_worth": net_worth,
            "annual_savings": annual_savings if not is_retired else 0,
            "annual_expense": annual_expense,
            "education_outflow": goals["education"],
            "marriage_outflow": goals["marriage"],
            "car_outflow": goals["car"],
            "house_outflow": goals["house"],
            "healthcare_loading": healthcare_loading,
        })

    df = pd.DataFrame(rows)
    return df, shortfall_age


def closed_form_required_corpus(inp: Inputs) -> float:
    """
    Sanity-check: required corpus at retirement using the standard real-return
    growing annuity formula, ignoring discrete goal outflows. This is used to
    cross-validate the year-by-year simulation and catch logic errors.
    """
    n = inp.life_expectancy - inp.retirement_age
    if n <= 0:
        return 0.0

    real_return = (1 + inp.post_retirement_return) / (1 + inp.general_inflation) - 1

    # first year retirement expense, projected from today's expenses
    years_to_retirement = inp.retirement_age - inp.current_age
    first_year_expense = (inp.monthly_expenses * 12) * (1 + inp.general_inflation) ** years_to_retirement

    if abs(real_return) < 1e-6:
        required = first_year_expense * n
    else:
        required = first_year_expense * (1 - (1 / (1 + real_return)) ** n) / real_return

    # gross up for tax drag
    required = required / (1 - inp.capital_gains_tax_rate)
    return required


def run_monte_carlo(inp: Inputs, n_sims: int = 500, return_vol_pre: float = 0.16, return_vol_post: float = 0.08):
    """
    Monte Carlo simulation of the same year-by-year engine with randomized
    annual returns (normal distribution around the assumed mean), to estimate
    probability of corpus survival to life expectancy and a distribution of
    legacy outcomes.
    """
    final_legacies = []
    survival_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        net_worth = inp.net_worth_liquid
        annual_savings = inp.monthly_savings * 12
        annual_expense = inp.monthly_expenses * 12
        survived = True

        for age in range(inp.current_age, inp.life_expectancy + 1):
            is_retired = age >= inp.retirement_age
            mean_ret = inp.post_retirement_return if is_retired else inp.pre_retirement_return
            vol = return_vol_post if is_retired else return_vol_pre
            ret_rate = rng.normal(mean_ret, vol)

            goals = goal_outflow_for_year(age, inp)
            goal_total = sum(goals.values())

            emi = inp.existing_emi_monthly * 12 if age < inp.emi_end_age else 0.0
            parent_support = inp.parents_support_monthly * 12 if age < inp.parents_support_end_age else 0.0

            healthcare_loading = 0.0
            if age >= 60:
                years_post_60 = age - 60
                healthcare_loading = (annual_expense * 0.08) * (1 + inp.healthcare_inflation) ** years_post_60

            if not is_retired:
                net_worth = net_worth * (1 + ret_rate) + annual_savings - goal_total - emi - parent_support - healthcare_loading
                annual_savings *= (1 + inp.income_growth_rate)
                annual_expense *= (1 + inp.general_inflation)
            else:
                withdrawal_needed = annual_expense + healthcare_loading + goal_total + parent_support + emi
                withdrawal_grossed_up = withdrawal_needed / (1 - inp.capital_gains_tax_rate)

                if net_worth <= 0:
                    net_worth = 0.0
                    survived = False
                else:
                    net_worth = net_worth * (1 + ret_rate) - withdrawal_grossed_up
                    if net_worth < 0:
                        net_worth = 0.0
                        survived = False

                annual_expense *= (1 + inp.general_inflation)

        final_legacies.append(max(net_worth, 0))
        if survived:
            survival_count += 1

    return np.array(final_legacies), survival_count / n_sims


# ──────────────────────────────────────────────────────────────────────────
# UI — SIDEBAR INPUTS
# ──────────────────────────────────────────────────────────────────────────

st.title("🏛️ Legacy Wealth Calculator")
st.caption("Retirement readiness and legacy planning — for client advisory use")

with st.sidebar:
    st.header("Client Profile")
    current_age = st.number_input("Current age", 18, 70, 40)
    retirement_age = st.number_input("Target retirement age", current_age + 1, 75, 60)
    life_expectancy = st.number_input("Life expectancy", retirement_age + 1, 100, 85)

    st.header("Net Worth & Cash Flow")
    net_worth_liquid = st.number_input("Liquid / investable net worth (₹)", 0, value=5_00_00_000, step=1_00_000, format="%d")
    net_worth_illiquid = st.number_input("Illiquid assets — house, etc. (₹, excluded from retirement corpus)", 0, value=2_00_00_000, step=1_00_000, format="%d")
    monthly_savings = st.number_input("Current monthly savings (₹)", 0, value=2_00_000, step=10_000, format="%d")
    monthly_expenses = st.number_input("Current monthly expenses (₹)", 0, value=3_00_000, step=10_000, format="%d")

    st.header("Return & Inflation Assumptions")
    pre_retirement_return = st.slider("Pre-retirement return (%)", 4.0, 16.0, 12.0) / 100
    post_retirement_return = st.slider("Post-retirement return (%)", 3.0, 12.0, 7.5) / 100
    income_growth_rate = st.slider("Annual savings growth (%)", 0.0, 15.0, 10.0) / 100
    general_inflation = st.slider("General inflation (%)", 2.0, 10.0, 6.0) / 100
    education_inflation = st.slider("Education inflation (%)", 4.0, 16.0, 10.0) / 100
    car_inflation = st.slider("Car cost inflation (%)", 2.0, 12.0, 5.0) / 100
    healthcare_inflation = st.slider("Healthcare inflation (post-60, %)", 4.0, 18.0, 4.0) / 100
    capital_gains_tax_rate = st.slider("Effective tax drag on withdrawals (%)", 0.0, 30.0, 10.0) / 100

    st.header("Life Goals")
    owns_house = st.checkbox("Already owns primary residence", value=True)
    house_target_cost_today = 0
    house_purchase_age = current_age
    if not owns_house:
        house_target_cost_today = st.number_input("Target house cost, today's value (₹)", 0, value=1_00_00_000, step=5_00_000, format="%d")
        house_purchase_age = st.number_input("Planned purchase age", current_age, retirement_age, current_age + 3)

    car_cost_today = st.number_input("Car cost, today's value (₹)", 0, value=30_00_000, step=1_00_000, format="%d")
    car_upgrade_every_n_years = st.number_input("Upgrade car every N years (0 = never)", 0, 20, 6)

    st.header("Other Obligations")
    existing_emi_monthly = st.number_input("Existing EMI, monthly (₹)", 0, value=0, step=5_000, format="%d")
    emi_end_age = current_age
    if existing_emi_monthly > 0:
        emi_end_age = st.number_input("EMI ends at age", current_age, 80, current_age + 10)

    parents_support_monthly = st.number_input("Monthly support to parents (₹)", 0, value=50_000, step=5_000, format="%d")
    parents_support_end_age = current_age
    if parents_support_monthly > 0:
        parents_support_end_age = st.number_input("Parent support ends at age", current_age, 90, 55)

    st.header("Children")
    num_children = st.number_input("Number of children", 0, 6, 2)
    default_ages = [12, 9]
    children = []
    for i in range(num_children):
        with st.expander(f"Child {i+1}", expanded=(i == 0)):
            name = st.text_input(f"Name", value=f"Child {i+1}", key=f"name_{i}")
            default_age = default_ages[i] if i < len(default_ages) else 5
            c_age = st.number_input(f"Current age", 0, 25, default_age, key=f"age_{i}")
            edu_cost = st.number_input(f"Education cost, today's value (₹, at age 18)", 0, value=2_00_00_000, step=1_00_000, format="%d", key=f"edu_{i}")
            marriage_cost = st.number_input(f"Marriage cost, today's value (₹, at age 25)", 0, value=1_00_00_000, step=1_00_000, format="%d", key=f"marriage_{i}")
            children.append(Child(name=name, current_age=c_age, education_cost_today=edu_cost, marriage_cost_today=marriage_cost))

    run_button = st.button("Calculate", type="primary", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────
# RUN & DISPLAY
# ──────────────────────────────────────────────────────────────────────────

if run_button or "has_run" in st.session_state:
    st.session_state["has_run"] = True

    inp = Inputs(
        current_age=current_age, retirement_age=retirement_age, life_expectancy=life_expectancy,
        net_worth_liquid=net_worth_liquid, net_worth_illiquid=net_worth_illiquid,
        monthly_savings=monthly_savings, monthly_expenses=monthly_expenses,
        pre_retirement_return=pre_retirement_return, post_retirement_return=post_retirement_return,
        income_growth_rate=income_growth_rate, general_inflation=general_inflation,
        education_inflation=education_inflation, car_inflation=car_inflation,
        healthcare_inflation=healthcare_inflation,
        owns_house=owns_house, house_target_cost_today=house_target_cost_today, house_purchase_age=house_purchase_age,
        car_cost_today=car_cost_today, car_upgrade_every_n_years=car_upgrade_every_n_years,
        children=children,
        existing_emi_monthly=existing_emi_monthly, emi_end_age=emi_end_age,
        parents_support_monthly=parents_support_monthly, parents_support_end_age=parents_support_end_age,
        capital_gains_tax_rate=capital_gains_tax_rate,
    )

    df, shortfall_age = run_deterministic_projection(inp)
    required_corpus_check = closed_form_required_corpus(inp)

    net_worth_at_retirement = df.loc[df["age"] == retirement_age, "net_worth"].values[0]
    legacy_nominal = df.loc[df["age"] == life_expectancy, "net_worth"].values[0]
    years_to_legacy = life_expectancy - current_age
    legacy_real = legacy_nominal / (1 + general_inflation) ** years_to_legacy

    with st.spinner("Running 500 simulated market paths..."):
        legacies, survival_prob = run_monte_carlo(inp)

    # ── Top-line metrics ──
    st.subheader("Headline Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net worth at retirement", f"₹{net_worth_at_retirement/1e7:,.2f} Cr")
    col2.metric("Legacy at life expectancy (nominal)", f"₹{legacy_nominal/1e7:,.2f} Cr")
    col3.metric("Probability corpus survives to life expectancy", f"{survival_prob*100:.0f}%")
    if children:
        per_child = max(legacy_nominal, 0) / len(children)
        child_breakdown = ", ".join([f"{c.name}: ₹{per_child/1e7:,.2f} Cr" for c in children])
        col4.metric("Legacy split per child (nominal)", f"₹{per_child/1e7:,.2f} Cr each")
    else:
        col4.metric("Legacy split per child", "No children added")

    if children and len(children) > 1:
        st.caption("Per-child split (equal): " + ", ".join([f"**{c.name}**: ₹{max(legacy_nominal,0)/len(children)/1e7:,.2f} Cr" for c in children]))

    if shortfall_age:
        st.error(f"⚠️ Shortfall detected: corpus is projected to run out at age {shortfall_age}, before life expectancy ({life_expectancy}). The retirement plan as structured is not sustainable — consider raising savings, delaying retirement, or reducing planned expenses.")
    else:
        st.success("✅ Corpus survives through life expectancy with no shortfall in the base-case projection.")

    # ── Sanity check ──
    with st.expander("🔍 Sanity check: closed-form vs. simulation"):
        st.write(f"**Closed-form required corpus at retirement (ignoring discrete goal outflows):** ₹{required_corpus_check/1e7:,.2f} Cr")
        st.write(f"**Simulated net worth at retirement (includes goal outflows up to that point):** ₹{net_worth_at_retirement/1e7:,.2f} Cr")
        st.caption("These won't match exactly since the closed-form ignores discrete goals like education/marriage/car, but they should be in the same order of magnitude. A large unexplained divergence signals a modeling error — check return assumptions and inflation compounding.")

    # ── Chart: net worth trajectory ──
    st.subheader("Net Worth Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["age"], y=df["net_worth"]/1e7, mode="lines", name="Net Worth (₹ Cr)",
                              line=dict(color="#1f4e79", width=3)))
    fig.add_vline(x=retirement_age, line_dash="dash", line_color="orange", annotation_text="Retirement")
    if shortfall_age:
        fig.add_vline(x=shortfall_age, line_dash="dash", line_color="red", annotation_text="Shortfall")
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(xaxis_title="Age", yaxis_title="Net Worth (₹ Crore)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    # ── Goal outflow breakdown ──
    st.subheader("Goal Outflows Over Time")
    goal_cols = ["education_outflow", "marriage_outflow", "car_outflow", "house_outflow"]
    goal_df = df[["age"] + goal_cols].copy()
    goal_df = goal_df[(goal_df[goal_cols] > 0).any(axis=1)]
    if not goal_df.empty:
        fig2 = go.Figure()
        for col, label, color in zip(goal_cols, ["Education", "Marriage", "Car", "House"],
                                       ["#2a9d8f", "#e76f51", "#e9c46a", "#264653"]):
            fig2.add_trace(go.Bar(x=goal_df["age"], y=goal_df[col]/1e5, name=label, marker_color=color))
        fig2.update_layout(barmode="stack", xaxis_title="Age", yaxis_title="Outflow (₹ Lakh)", height=350)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No discrete goal outflows configured.")

    # ── Monte Carlo ──
    st.subheader("Monte Carlo: Robustness Check")
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_col1.metric("Probability corpus survives to life expectancy", f"{survival_prob*100:.0f}%")
    mc_col2.metric("Median legacy (nominal)", f"₹{np.median(legacies)/1e7:,.2f} Cr")
    mc_col3.metric("10th percentile legacy (downside case)", f"₹{np.percentile(legacies,10)/1e7:,.2f} Cr")

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=legacies/1e7, nbinsx=40, marker_color="#1f4e79"))
    fig3.update_layout(xaxis_title="Legacy at life expectancy (₹ Crore)", yaxis_title="Number of simulations", height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.caption("Monte Carlo randomizes annual returns around the assumed mean (16% volatility pre-retirement, 8% post-retirement) across 500 simulated paths, to capture sequence-of-returns risk that a single fixed-return projection hides.")

    # ── Full data table ──
    with st.expander("📊 Full year-by-year projection"):
        display_df = df.copy()
        for c in ["net_worth", "annual_savings", "annual_expense", "education_outflow",
                  "marriage_outflow", "car_outflow", "house_outflow", "healthcare_loading"]:
            display_df[c] = display_df[c].round(0).astype("int64")
        st.dataframe(display_df, use_container_width=True, height=400)

    st.caption("This tool is for illustrative advisory purposes. All projections are based on stated assumptions and are not guarantees of future performance.")

else:
    st.info("Set the client's parameters in the sidebar and click **Calculate** to generate the legacy wealth projection.")
