import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Panic Tax Calculator – HNI", layout="wide")

# Custom CSS for structural layout and HNI theme styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px 20px;
        border-left: 5px solid #ccc;
        margin-bottom: 10px;
    }
    .metric-card-steady { border-left-color: #10B981; }
    .metric-card-panic { border-left-color: #EF4444; }
    .metric-card-opp { border-left-color: #3B82F6; }
    .metric-label { font-size: 0.9rem; color: #6c757d; font-weight: 500; }
    .metric-value { font-size: 1.6rem; font-weight: bold; color: #212529; }
    </style>
""", unsafe_allow_html=True)

# Historical Event Library
EVENT_LIBRARY = {
    "2008 GFC Bottom": "2009-03-31",
    "2011 Eurozone Crisis Bottom": "2011-11-30",
    "2016 Demonetization Dip": "2016-11-30",
    "2018 NBFC Crisis Bottom": "2018-10-31",
    "2020 COVID-19 Bottom": "2020-03-31",
    "2022 Russia-Ukraine Dip": "2022-06-30",
    "2024 Election Result Dip": "2024-06-30"
}

# Initialize behavioral events ledger state if empty
if 'timeline_events' not in st.session_state:
    st.session_state.timeline_events = [
        {
            "Event Name": "2020 COVID-19 Bottom",
            "Date": "2020-03-31",
            "Action Type": "Pause SIP (Months)",
            "Value": 24.0
        }
    ]

# ──────────────────────── 0. Indian Number System Formatter ────────────────────────
def format_indian_currency(number):
    if number is None:
        return "₹ 0"
    try:
        is_negative = number < 0
        number = abs(number)
        num_str = f"{number:.0f}"
        if len(num_str) <= 3:
            formatted = num_str
        else:
            last_three = num_str[-3:]
            remaining = num_str[:-3]
            groups = []
            while len(remaining) > 0:
                groups.append(remaining[-2:])
                remaining = remaining[:-2]
            groups.reverse()
            formatted = ",".join(groups) + "," + last_three
        return f"-₹ {formatted}" if is_negative else f"₹ {formatted}"
    except Exception:
        return str(number)

def format_indian_no_symbol(number):
    if number is None:
        return "0"
    try:
        is_negative = number < 0
        number = abs(number)
        num_str = f"{number:.0f}"
        if len(num_str) <= 3:
            formatted = num_str
        else:
            last_three = num_str[-3:]
            remaining = num_str[:-3]
            groups = []
            while len(remaining) > 0:
                groups.append(remaining[-2:])
                remaining = remaining[:-2]
            groups.reverse()
            formatted = ",".join(groups) + "," + last_three
        return f"-{formatted}" if is_negative else formatted
    except Exception:
        return str(number)

# ──────────────────────── 1. Data Engine ────────────────────────
@st.cache_data(ttl=3600)
def get_nifty_price_series():
    try:
        nifty = yf.download("^NSEI", start="1999-01-01", progress=False)
        if nifty.empty:
            raise ValueError("Empty data")
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.get_level_values(0)
        
        monthly = nifty['Close'].resample('ME').last().ffill().dropna()
        if monthly.empty:
            raise ValueError("No close data")
        
        if isinstance(monthly, pd.DataFrame):
            monthly = monthly.iloc[:, 0]
        return monthly
    except Exception:
        dates = pd.date_range("1999-01-31", datetime.now(), freq='ME')
        np.random.seed(42)
        rets = np.random.normal(0.011, 0.06, len(dates)-1)
        
        peak_2008 = np.where(dates >= '2008-01-01')[0][0]
        for i in range(peak_2008+1, peak_2008+7):
            rets[i] = -0.12
        covid = np.where(dates >= '2020-01-31')[0][0]
        rets[covid] = -0.03
        rets[covid+1] = -0.35
        
        prices = [1000.0]
        for r in rets:
            prices.append(prices[-1] * (1 + r))
        return pd.Series(prices, index=dates)

def extend_series(series, target_date, annual_return=0.12):
    last_date = series.index[-1]
    if target_date <= last_date:
        return series
    future_dates = pd.date_range(last_date + pd.DateOffset(months=1), target_date, freq='ME')
    monthly_ret = (1 + annual_return) ** (1/12) - 1
    last_val = float(series.iloc[-1])
    new_vals = []
    for _ in future_dates:
        last_val *= (1 + monthly_ret)
        new_vals.append(last_val)
    return pd.concat([series, pd.Series(new_vals, index=future_dates)])

# ──────────────────────── 2. Multi-Event FIFO Simulation Engine ────────────────────────
def simulate_multi_events(prices, start_idx, end_idx, monthly_sip,
                          existing_corpus=0, events_list=None,
                          reinvest_paused_cash=False):
    if events_list is None:
        events_list = []

    portfolio_queue = []
    
    if existing_corpus > 0:
        init_nav = float(prices.iloc[start_idx])
        init_units = existing_corpus / init_nav
        portfolio_queue.append([init_units, init_nav, prices.index[start_idx]])

    withdrawal_cash = 0.0
    paused_sip_cash = 0.0
    total_invested = existing_corpus
    tax_paid = 0.0
    wealth_history = []

    active_sip_pauses = []

    events_by_date = {}
    for ev in events_list:
        ev_date = pd.Timestamp(ev["date"]).normalize()
        events_by_date.setdefault(ev_date, []).append(ev)

    for i in range(start_idx, end_idx + 1):
        current_date = prices.index[i].normalize()
        nav = float(prices.iloc[i])

        active_sip_pauses = [p for p in active_sip_pauses if p["resume_date"] > current_date]

        day_events = events_by_date.get(current_date, [])
        
        withdrawals = [e for e in day_events if "Withdrawal" in e["action"]]
        pauses = [e for e in day_events if "Pause" in e["action"]]
        lumpsums = [e for e in day_events if "Lumpsum" in e["action"]]

        for p in pauses:
            pause_months_count = int(p["value"])
            resume_date = current_date + pd.DateOffset(months=pause_months_count)
            active_sip_pauses.append({"resume_date": resume_date.normalize()})

        for w in withdrawals:
            frac = min(1.0, max(0.0, float(w["value"]) / 100.0))
            if frac > 0:
                total_units_held = sum(batch[0] for batch in portfolio_queue)
                units_to_redeem = total_units_held * frac
                
                redeemed_remaining = units_to_redeem
                gross_redemption_value = 0.0
                local_tax_paid = 0.0
                redeemed_cost_basis = 0.0
                next_portfolio_queue = []

                for batch in portfolio_queue:
                    b_units, b_price, b_date = batch
                    if redeemed_remaining <= 0:
                        next_portfolio_queue.append(batch)
                        continue

                    if b_units <= redeemed_remaining:
                        redeemed_units = b_units
                        redeemed_remaining -= b_units
                    else:
                        redeemed_units = redeemed_remaining
                        next_portfolio_queue.append([b_units - redeemed_units, b_price, b_date])
                        redeemed_remaining = 0.0

                    batch_gross = redeemed_units * nav
                    batch_cost = redeemed_units * b_price
                    redeemed_cost_basis += batch_cost
                    batch_gains = max(0.0, batch_gross - batch_cost)
                    
                    holding_days = (current_date - b_date).days
                    is_ltcg = holding_days > 365
                    tax_rate = 0.125 if is_ltcg else 0.20
                    
                    local_tax_paid += batch_gains * tax_rate
                    gross_redemption_value += batch_gross

                portfolio_queue = next_portfolio_queue
                tax_paid += local_tax_paid
                
                net_proceeds = gross_redemption_value - local_tax_paid
                withdrawal_cash += net_proceeds

                total_invested = max(0.0, total_invested - redeemed_cost_basis)

        for l in lumpsums:
            amt = float(l["value"])
            if amt > 0:
                units_bought = amt / nav
                portfolio_queue.append([units_bought, nav, current_date])
                total_invested += amt

        is_sip_paused = len(active_sip_pauses) > 0
        if is_sip_paused:
            paused_sip_cash += monthly_sip
        else:
            if reinvest_paused_cash and paused_sip_cash > 0:
                units_from_cash = paused_sip_cash / nav
                portfolio_queue.append([units_from_cash, nav, current_date])
                total_invested += paused_sip_cash
                paused_sip_cash = 0.0

            units_bought = monthly_sip / nav
            portfolio_queue.append([units_bought, nav, current_date])
            total_invested += monthly_sip

        current_units = sum(batch[0] for batch in portfolio_queue)
        wealth_history.append((current_units * nav) + withdrawal_cash + paused_sip_cash)

    final_units = sum(batch[0] for batch in portfolio_queue)
    final_wealth = (final_units * float(prices.iloc[end_idx])) + withdrawal_cash + paused_sip_cash

    return {
        'final_wealth': final_wealth,
        'wealth_history': wealth_history,
        'total_invested': total_invested,
        'tax_paid': tax_paid
    }

# ──────────────────────── 3. Main Application Layout ────────────────────────
with st.sidebar:
    st.header("👤 Client Details")
    client_name = st.text_input("Client Name", "Mr. Sharma")
    current_age = st.number_input("Current Age (Years)", min_value=1, max_value=59, value=35, step=1)

    years_to_60 = 60 - current_age
    calculated_end_date = datetime.today() + pd.DateOffset(years=years_to_60)
    end_date = calculated_end_date.date()

    st.header("⏳ Horizon Parameters")
    start_date = st.date_input("Investment Start Date", value=datetime(2015, 1, 1))
    
    st.info(f"Target Retirement Date (Age 60): **{end_date.strftime('%d-%b-%Y')}**")
    
    monthly_sip = st.number_input("Monthly SIP Amount (₹)", 0, 10000000, 100000, step=10000)
    st.caption(f"Formatted Monthly SIP: **{format_indian_currency(monthly_sip)}**")
    
    existing_corpus = st.number_input("Starting Portfolio Value (₹)", 0, 1000000000, 5000000, step=100000)
    st.caption(f"Formatted Starting Portfolio: **{format_indian_currency(existing_corpus)}**")
    
    future_cagr = st.slider("Expected Long-term CAGR (%)", 5.0, 20.0, 12.0) / 100

    reinvest_paused_cash = st.checkbox(
        "Auto-Reinvest Paused SIP Cash",
        value=False,
        help="If checked, accumulated SIP cash during pauses will be reinvested into the market when the pause ends. If unchecked, it remains as idle cash earning 0%."
    )

    st.header("🔄 Add Behavioral Stress Action")
    st.markdown("Build your timeline by picking predefined historical dip points below:")

    selected_historical_event = st.selectbox("Select Historical Dip", list(EVENT_LIBRARY.keys()))
    selected_action = st.selectbox("Reaction Action", ["Pause SIP (Months)", "Partial Withdrawal (%)", "Opportunistic Lumpsum (Rs.)"])

    if "Pause" in selected_action:
        val_help = "Months to pause"
        default_val = 24.0
    elif "Withdrawal" in selected_action:
        val_help = "Percentage to redeem (10-100)"
        default_val = 30.0
    else:
        val_help = "Absolute Rs. amount to invest"
        default_val = 500000.0

    action_value = st.number_input(f"Value ({val_help})", min_value=0.0, value=default_val, step=1.0 if "Pause" in selected_action else 10000.0)
    
    if "Lumpsum" in selected_action:
        st.caption(f"Formatted Lumpsum: **{format_indian_currency(action_value)}**")

    if st.button("➕ Add Action to Timeline", use_container_width=True):
        st.session_state.timeline_events.append({
            "Event Name": selected_historical_event,
            "Date": EVENT_LIBRARY[selected_historical_event],
            "Action Type": selected_action,
            "Value": action_value
        })
        st.toast(f"Added {selected_action} on {selected_historical_event} to ledger.", icon="✅")

    if st.session_state.timeline_events:
        st.markdown("---")
        st.subheader("📋 Active Behavioral Timeline")
        
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        to_delete = None
        
        for idx, ev in enumerate(st.session_state.timeline_events):
            ev_date = pd.Timestamp(ev['Date'])
            is_valid = start_ts <= ev_date <= end_ts
            status_indicator = "✅" if is_valid else "⚠️ (Out)"
            
            col_text, col_btn = st.columns([0.8, 0.2])
            with col_text:
                if "Lumpsum" in ev['Action Type']:
                    formatted_val_str = format_indian_currency(ev['Value'])
                elif "Withdrawal" in ev['Action Type']:
                    formatted_val_str = f"{ev['Value']}%"
                else:
                    formatted_val_str = f"{ev['Value']} months"

                st.caption(
                    f"**{idx + 1}. {ev['Event Name']}** ({ev['Date']}) {status_indicator}  \n"
                    f"Action: {ev['Action Type']} | Value: {formatted_val_str}"
                )
            with col_btn:
                if st.button("🗑️", key=f"del_{idx}", help="Remove this single action"):
                    to_delete = idx

        if to_delete is not None:
            st.session_state.timeline_events.pop(to_delete)
            st.toast("Selected behavioral action removed.", icon="🗑️")
            st.rerun()
        
        st.markdown("---")
        if st.button("❌ Reset Entire Timeline", type="secondary", use_container_width=True):
            st.session_state.timeline_events = []
            st.rerun()

# ──────────────────────── 4. Real-time Calculation Engine ────────────────────────
price_series_raw = get_nifty_price_series()
extended_prices = extend_series(price_series_raw, pd.Timestamp(end_date), annual_return=future_cagr)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

active_prices = extended_prices[(extended_prices.index >= start_ts) & (extended_prices.index <= end_ts)]

if len(active_prices) < 2:
    st.error("Timeline boundary is insufficient. Expand the horizon range by adjusting your age or investment start date.")
    st.stop()

start_idx = 0
end_idx = len(active_prices) - 1

panic_events = []
opportunist_events = []
all_combined_events = []
has_out_of_horizon = False

for ev in st.session_state.timeline_events:
    dt_raw = pd.Timestamp(ev["Date"])
    
    if not (start_ts <= dt_raw <= end_ts):
        has_out_of_horizon = True
        continue
        
    idx_matched = active_prices.index.get_indexer([dt_raw], method='nearest')[0]
    dt_matched = active_prices.index[idx_matched]
    
    val = float(ev["Value"])
    act = ev["Action Type"]
    
    if "Lumpsum" in act:
        opp_entry = {"date": dt_matched, "action": "Lumpsum", "value": val}
        opportunist_events.append(opp_entry)
        all_combined_events.append(opp_entry)
    else:
        panic_entry = {"date": dt_matched, "action": "Pause" if "Pause" in act else "Withdrawal", "value": val}
        panic_events.append(panic_entry)
        all_combined_events.append(panic_entry)

if has_out_of_horizon:
    st.sidebar.warning("⚠️ Some selected historical events reside outside your current Investment Horizon and were omitted from the projection.")

# Run Simulations
steady_out = simulate_multi_events(active_prices, start_idx, end_idx, monthly_sip, existing_corpus, reinvest_paused_cash=reinvest_paused_cash)
panic_out = simulate_multi_events(active_prices, start_idx, end_idx, monthly_sip, existing_corpus, 
                                  events_list=panic_events, reinvest_paused_cash=reinvest_paused_cash)
opp_out = simulate_multi_events(active_prices, start_idx, end_idx, monthly_sip, existing_corpus, 
                                events_list=opportunist_events, reinvest_paused_cash=reinvest_paused_cash)
combined_out = simulate_multi_events(active_prices, start_idx, end_idx, monthly_sip, existing_corpus, 
                                     events_list=all_combined_events, reinvest_paused_cash=reinvest_paused_cash)

# Extract core metrics
wealth_steady = steady_out['final_wealth']
wealth_panicker = panic_out['final_wealth']
wealth_opp = opp_out['final_wealth']
wealth_combined = combined_out['final_wealth']
tax_paid = combined_out['tax_paid']

# Calculate Panic Tax comparing Evolving steady state vs the client's actual behavioral path
panic_tax = wealth_steady - wealth_combined
panic_tax_pct = (panic_tax / wealth_steady * 100) if wealth_steady > 0 else 0.0

# Calculate target postponement delay based on the Composite behavioral journey
extra_years = 0.0
if wealth_combined < wealth_steady and monthly_sip > 0:
    monthly_cagr_rate = (1 + future_cagr) ** (1/12) - 1
    current_temp_val = wealth_combined
    for m in range(1, 1200):
        current_temp_val = current_temp_val * (1 + monthly_cagr_rate) + monthly_sip
        if current_temp_val >= wealth_steady:
            extra_years = m / 12
            break

# ──────────────────────── UI Reporting Dashboard ────────────────────────
st.subheader("📊 Strategic Projections Dashboard")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card metric-card-steady">
            <div class="metric-label">Steady Portfolio (No Reaction at Age 60)</div>
            <div class="metric-value">{format_indian_currency(wealth_steady)}</div>
            <p style="margin:5px 0 0 0; font-size:0.85rem; color:#6c757d;">Total Invested: {format_indian_currency(steady_out['total_invested'])}</p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card metric-card-panic">
            <div class="metric-label">Composite Journey (Your Path at Age 60)</div>
            <div class="metric-value">{format_indian_currency(wealth_combined)}</div>
            <p style="margin:5px 0 0 0; font-size:0.85rem; color:#EF4444;">Wealth Variance: {format_indian_currency(-panic_tax)}</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card metric-card-opp">
            <div class="metric-label">Pure Opportunistic Strategy</div>
            <div class="metric-value">{format_indian_currency(wealth_opp)}</div>
            <p style="margin:5px 0 0 0; font-size:0.85rem; color:#3B82F6;">Total Invested: {format_indian_currency(opp_out['total_invested'])}</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"### 💸 Net Panic Tax: <span style='color:#EF4444; font-weight:bold;'>{format_indian_currency(panic_tax)}</span>", unsafe_allow_html=True)
st.markdown(f"The structural behavior disruption cost represents **{panic_tax_pct:.1f}%** of the steady-state potential portfolio, permanently foregone.")

st.info(
    "💡 **Interpretation Note:** When a panic withdrawal occurs, your overall portfolio value "
    "does not instantly drop on that exact date. The withdrawn cash continues to be tracked as part of your total holdings. "
    "The real cost of the panic tax accumulates after the event, representing the opportunity cost of holding cash while "
    "the equity market recovers, along with any capital gains tax realized at the point of sale."
)

if tax_paid > 0:
    st.warning(f"🚨 **Realized Capital Gains Tax**: A total of **{format_indian_currency(tax_paid)}** was permanently locked-in as paid tax due to liquidation. This structural loss of capital is irreversible.")
else:
    st.info("ℹ️ **No Realized Capital Gains Tax**: Capital Gains Tax remains **₹ 0** because either (a) the selected behavioral timeline involves only SIP pauses rather than withdrawals, or (b) redemptions were offset by capital losses incurred at the crash bottom.")

luxury_equivalents = [
    (100000000, "a premium estate property"),
    (50000000, "a luxury apartment"),
    (10000000, "a real estate property downpayment"),
    (5000000, "a luxury vehicle"),
    (2500000, "a SUV"),
    (1000000, "a luxury timepiece portfolio")
]
for threshold, item in luxury_equivalents:
    if panic_tax >= threshold:
        st.info(f"💡 This lost value equates approximately to **{item}**.")
        break

if extra_years > 0:
    st.warning(f"⏳ **Investment Horizon Impact**: To restore the wealth gap, investment durations must continue for an additional **{extra_years:.1f} years** under assumptions.")

# ──────────────────────── Plotly Data Visualization ────────────────────────
st.subheader("📈 Wealth Path Tracking")

dates_formatted = active_prices.index.strftime('%Y-%m-%d')

# Generate customized ticks using the Indian numbering system
all_histories = (
    steady_out['wealth_history'] + 
    combined_out['wealth_history'] + 
    opp_out['wealth_history'] + 
    panic_out['wealth_history']
)
min_y = min(all_histories)
max_y = max(all_histories)

# Evenly spaced tick intervals
tick_vals = np.linspace(min_y, max_y, 7)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dates_formatted, 
    y=steady_out['wealth_history'], 
    name="Steady (No Reaction)", 
    line=dict(color='#10B981', width=2.5)
))

fig.add_trace(go.Scatter(
    x=dates_formatted, 
    y=combined_out['wealth_history'], 
    name="Your Composite Journey", 
    line=dict(color='#8B5CF6', width=2.5)
))

fig.add_trace(go.Scatter(
    x=dates_formatted, 
    y=opp_out['wealth_history'], 
    name="Pure Opportunistic (No Panic)", 
    line=dict(color='#3B82F6', dash='dot', width=2)
))

fig.add_trace(go.Scatter(
    x=dates_formatted, 
    y=panic_out['wealth_history'], 
    name="Pure Panic (No Lumpsum)", 
    line=dict(color='#EF4444', dash='dash', width=2)
))

# Stagger and vertical reference markings
y_min_val = min_y * 0.95
y_max_val = max_y * 1.05

for idx, ev in enumerate(st.session_state.timeline_events):
    dt_raw = pd.Timestamp(ev["Date"])
    if start_ts <= dt_raw <= end_ts:
        dt_str = dt_raw.strftime('%Y-%m-%d')
        fig.add_trace(go.Scatter(
            x=[dt_str, dt_str],
            y=[y_min_val, y_max_val],
            mode='lines',
            line=dict(color='#64748B', dash='dot', width=1.2),
            showlegend=False
        ))
        fig.add_annotation(
            x=dt_str, 
            y=y_max_val - (idx * (y_max_val * 0.05)),
            text=ev["Event Name"],
            showarrow=False,
            yshift=10,
            font=dict(color="#64748B", size=8)
        )

fig.update_layout(
    xaxis_title="Timeline Date", 
    yaxis_title="Portfolio Value (INR)", 
    hovermode="x unified", 
    template="plotly_white",
    margin=dict(l=75, r=15, t=30, b=45),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=9, color="#475569")
    )
)

fig.update_xaxes(
    type='date',
    tickformat='%b %Y',
    gridcolor="#F1F5F9",
    linecolor="#CBD5E1",
    tickfont=dict(color="#64748B", size=9)
)

fig.update_yaxes(
    gridcolor="#F1F5F9",
    linecolor="#CBD5E1",
    tickvals=tick_vals,
    ticktext=[format_indian_no_symbol(val) for val in tick_vals],
    tickfont=dict(color="#64748B", size=9)
)

st.plotly_chart(fig, use_container_width=True)