# Legacy Wealth Calculator

A retirement readiness and legacy planning tool built with Streamlit. Projects a client's net worth through retirement and decumulation, accounts for major life goals (children's education and marriage, car upgrades, home purchase, parental support), models healthcare cost escalation, and runs a Monte Carlo simulation to estimate the probability of corpus survival to life expectancy.

## Features

- Two-phase net worth projection: accumulation (pre-retirement) and decumulation (post-retirement), with separate return assumptions for each phase
- Goal-based outflows: children's education and marriage costs (inflated separately from general CPI), periodic car upgrades, one-time home purchase, parental support, existing EMI
- Healthcare cost escalation post-age 60
- Tax-drag-adjusted retirement withdrawals
- Built-in sanity check: closed-form annuity formula cross-validates the year-by-year simulation
- Monte Carlo simulation (500 paths) for probability of corpus survival and downside legacy estimates
- Legacy figure split per child

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

## Disclaimer

This tool is for illustrative advisory purposes only. All projections are based on user-supplied assumptions and are not guarantees of future performance.
