"""
Stock Analyzer - Web App (Streamlit)

Run locally:
    pip install streamlit yfinance pandas numpy ta plotly requests
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push app.py + requirements.txt to GitHub, connect at share.streamlit.io
    The yfinance rate-limit fix (fake browser session) is built in.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go
import plotly.express as px
import requests

# ---------------------------------------------------------------------------
# yfinance rate-limit fix: use a real browser session so Streamlit Cloud
# doesn't get blocked by Yahoo Finance
# ---------------------------------------------------------------------------

def make_yf_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(symbol):
    try:
        session = make_yf_session()
        t = yf.Ticker(symbol, session=session)
        info = t.info
        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            return {"error": f"No data found for '{symbol}'. Check the ticker."}
        return {
            "info":             info,
            "hist":             t.history(period="2y"),
            "income_stmt":      t.income_stmt,
            "quarterly_income": t.quarterly_income_stmt,
            "balance_sheet":    t.balance_sheet,
            "cashflow":         t.cashflow,
            "recommendations":  t.recommendations,
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer(ticker):
    try:
        session = make_yf_session()
        t = yf.Ticker(ticker, session=session)
        return t.info or {}
    except Exception:
        return {}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sg(d, *keys, default=None):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default

def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    return f"{x * 100:.2f}%"

def fmt_num(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    return f"{x:.{dec}f}"

def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "N/A"
    ax = abs(x)
    if ax >= 1e12: return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.2f}B"
    if ax >= 1e6:  return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"

def show_table(metrics, pct_keys=None, money_keys=None):
    pct_keys, money_keys = pct_keys or [], money_keys or []
    rows = []
    for k, v in metrics.items():
        if   k in pct_keys:   display = fmt_pct(v)
        elif k in money_keys: display = fmt_money(v)
        elif isinstance(v, float): display = fmt_num(v)
        else: display = str(v) if v is not None else "N/A"
        rows.append({"Metric": k, "Value": display})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

def cagr(series):
    series = series.dropna()
    if len(series) < 2: return None
    vals = series.values[::-1]
    s, e = vals[0], vals[-1]
    n = len(vals) - 1
    if s <= 0 or e <= 0 or n <= 0: return None
    return (e / s) ** (1 / n) - 1

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_valuation(info):
    return {
        "Trailing P/E":  sg(info, "trailingPE"),
        "Forward P/E":   sg(info, "forwardPE"),
        "PEG Ratio":     sg(info, "pegRatio", "trailingPegRatio"),
        "Price/Book":    sg(info, "priceToBook"),
        "Price/Sales":   sg(info, "priceToSalesTrailing12Months"),
        "EV/EBITDA":     sg(info, "enterpriseToEbitda"),
        "EV/Revenue":    sg(info, "enterpriseToRevenue"),
    }

def compute_growth(info, income_stmt):
    m = {
        "Revenue Growth (YoY)":            sg(info, "revenueGrowth"),
        "Earnings Growth (YoY)":           sg(info, "earningsGrowth"),
        "Quarterly Earnings Growth (YoY)": sg(info, "earningsQuarterlyGrowth"),
    }
    try:
        rev = income_stmt.loc["Total Revenue"]
        m["Revenue 3-4Y CAGR"] = cagr(rev)
    except Exception: m["Revenue 3-4Y CAGR"] = None
    try:
        ni = income_stmt.loc["Net Income"]
        m["Net Income 3-4Y CAGR"] = cagr(ni)
    except Exception: m["Net Income 3-4Y CAGR"] = None
    return m

def compute_profitability(info):
    return {
        "Gross Margin":    sg(info, "grossMargins"),
        "Operating Margin":sg(info, "operatingMargins"),
        "Net Margin":      sg(info, "profitMargins"),
        "ROE":             sg(info, "returnOnEquity"),
        "ROA":             sg(info, "returnOnAssets"),
    }

def compute_health(info):
    return {
        "Debt/Equity":   sg(info, "debtToEquity"),
        "Current Ratio": sg(info, "currentRatio"),
        "Quick Ratio":   sg(info, "quickRatio"),
        "Free Cash Flow":sg(info, "freeCashflow"),
    }

def compute_future(info):
    price  = sg(info, "currentPrice", "regularMarketPrice")
    target = sg(info, "targetMeanPrice")
    upside = (target - price) / price if price and target else None
    return {
        "Current Price":          price,
        "Analyst Target Mean":    target,
        "Analyst Target High":    sg(info, "targetHighPrice"),
        "Analyst Target Low":     sg(info, "targetLowPrice"),
        "Implied Upside":         upside,
        "Recommendation":         sg(info, "recommendationKey"),
        "# Analyst Opinions":     sg(info, "numberOfAnalystOpinions"),
        "Forward EPS":            sg(info, "forwardEps"),
        "Trailing EPS":           sg(info, "trailingEps"),
    }

def compute_technicals(hist):
    if hist is None or hist.empty: return {}
    close   = hist["Close"]
    current = close.iloc[-1]
    ma50    = close.rolling(50).mean().iloc[-1]  if len(close) >= 50  else None
    ma200   = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    rsi     = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1] if len(close) > 14 else None
    macd_o  = ta.trend.MACD(close)
    macd    = macd_o.macd().iloc[-1]
    msig    = macd_o.macd_signal().iloc[-1]
    h52, l52 = close[-252:].max(), close[-252:].min()
    return {
        "Current Price":   current,
        "50-Day MA":       ma50,
        "200-Day MA":      ma200,
        "RSI (14)":        rsi,
        "MACD":            macd,
        "MACD Signal":     msig,
        "52-Week High":    h52,
        "52-Week Low":     l52,
        "% From 52W High": (current - h52) / h52 if h52 else None,
        "% From 52W Low":  (current - l52) / l52 if l52 else None,
    }

# ---------------------------------------------------------------------------
# Scoring  (-1 = undervalued/bullish, +1 = overvalued/bearish)
# ---------------------------------------------------------------------------

def score_val(m):
    s = []
    pe = m.get("Trailing P/E")
    if pe:  s.append(-1 if pe < 15 else (1 if pe > 30 else 0))
    peg = m.get("PEG Ratio")
    if peg: s.append(-1 if peg < 1 else (1 if peg > 2 else 0))
    pb = m.get("Price/Book")
    if pb:  s.append(-1 if pb < 1 else (1 if pb > 5 else 0))
    ev = m.get("EV/EBITDA")
    if ev:  s.append(-1 if ev < 8 else (1 if ev > 18 else 0))
    return np.mean(s) if s else 0

def score_growth(m):
    s = []
    for k, hi, lo in [("Revenue Growth (YoY)", 0.15, 0),
                       ("Earnings Growth (YoY)", 0.15, 0),
                       ("Revenue 3-4Y CAGR", 0.10, 0)]:
        v = m.get(k)
        if v is not None: s.append(-1 if v > hi else (1 if v < lo else 0))
    return np.mean(s) if s else 0

def score_profit(m):
    s = []
    nm = m.get("Net Margin")
    if nm is not None: s.append(-1 if nm > 0.15 else (1 if nm < 0.05 else 0))
    roe = m.get("ROE")
    if roe is not None: s.append(-1 if roe > 0.15 else (1 if roe < 0.05 else 0))
    return np.mean(s) if s else 0

def score_future(m):
    s = []
    up = m.get("Implied Upside")
    if up is not None: s.append(-1 if up > 0.10 else (1 if up < -0.05 else 0))
    rec = str(m.get("Recommendation", "")).lower()
    if rec in ("strong_buy", "buy"): s.append(-1)
    elif rec in ("sell", "strong_sell"): s.append(1)
    elif rec: s.append(0)
    return np.mean(s) if s else 0

def score_tech(m):
    s = []
    rsi = m.get("RSI (14)")
    if rsi is not None: s.append(-1 if rsi < 30 else (1 if rsi > 70 else 0))
    price, ma50, ma200 = m.get("Current Price"), m.get("50-Day MA"), m.get("200-Day MA")
    if price and ma50:  s.append(-0.5 if price < ma50  else 0.5)
    if price and ma200: s.append(-0.5 if price < ma200 else 0.5)
    macd, msig = m.get("MACD"), m.get("MACD Signal")
    if macd is not None and msig is not None:
        s.append(-0.5 if macd > msig else 0.5)
    return np.mean(s) if s else 0

WEIGHTS = {"Valuation": 0.35, "Growth": 0.20, "Profitability": 0.15,
           "Future Outlook": 0.15, "Technicals": 0.15}

def get_verdict(scores):
    total = sum(scores[c] * WEIGHTS[c] for c in WEIGHTS)
    if   total <= -0.3: return total, "UNDERVALUED",   "green"
    elif total >=  0.3: return total, "OVERVALUED",    "red"
    else:               return total, "FAIRLY VALUED",  "orange"

# ---------------------------------------------------------------------------
# DCF Calculator
# ---------------------------------------------------------------------------

def run_dcf(fcf, revenue, growth_rate, terminal_growth, discount_rate, years, shares, net_debt):
    base = fcf if (fcf and fcf > 0) else (revenue * 0.05 if revenue else None)
    if not base or not shares or shares == 0: return None, None, None

    cf_rows = []
    for y in range(1, years + 1):
        cf = base * ((1 + growth_rate) ** y)
        pv = cf / ((1 + discount_rate) ** y)
        cf_rows.append({"Year": f"Year {y}", "FCF ($M)": cf / 1e6, "PV ($M)": pv / 1e6})

    t_fcf = base * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    tv    = t_fcf / (discount_rate - terminal_growth)
    tv_pv = tv / ((1 + discount_rate) ** years)

    equity_val = (sum(r["PV ($M)"] for r in cf_rows) * 1e6 + tv_pv) - (net_debt or 0)
    return equity_val / shares, cf_rows, tv_pv / 1e6

# ---------------------------------------------------------------------------
# Peer comparison metrics
# ---------------------------------------------------------------------------

PEER_KEYS = {
    "P/E":          ("trailingPE",                     False),
    "Forward P/E":  ("forwardPE",                      False),
    "P/B":          ("priceToBook",                    False),
    "EV/EBITDA":    ("enterpriseToEbitda",             False),
    "Net Margin":   ("profitMargins",                  True),
    "Gross Margin": ("grossMargins",                   True),
    "ROE":          ("returnOnEquity",                 True),
    "Debt/Equity":  ("debtToEquity",                   False),
    "Rev Growth":   ("revenueGrowth",                  True),
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock Analyzer", page_icon="📊", layout="wide")
st.title("📊 Stock Analyzer")
st.caption("Fundamentals · Growth · Profitability · DCF Valuation · Peer Comparison · Technicals")

# --- Ticker input ---
col_in, col_btn = st.columns([3, 1])
with col_in:
    symbol = st.text_input("Ticker Symbol", value=st.session_state.get("last_symbol", "AAPL"),
                           label_visibility="collapsed").strip().upper()
with col_btn:
    analyze = st.button("Analyze", type="primary", use_container_width=True)

# Fetch data only when Analyze is clicked; store in session_state so sliders
# and other widgets don't trigger a fresh fetch / wipe the results.
if analyze and symbol:
    with st.spinner(f"Fetching data for {symbol}..."):
        data = fetch_all(symbol)
    if "error" in data:
        st.error(data["error"])
        st.stop()
    st.session_state["data"]        = data
    st.session_state["last_symbol"] = symbol
    st.session_state["peer_infos"]  = None   # reset peers when new ticker analyzed

# Only render results if we have data stored
if "data" not in st.session_state:
    st.info("Enter a ticker symbol above and click **Analyze** to get started.")
    st.stop()

data    = st.session_state["data"]
symbol  = st.session_state["last_symbol"]
info    = data["info"]
hist    = data["hist"]
inc     = data["income_stmt"]
bal     = data["balance_sheet"]
cf_stmt = data["cashflow"]

# --- Header ---
name  = info.get("longName", symbol)
price = sg(info, "currentPrice", "regularMarketPrice")
st.header(f"{name} ({symbol})")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Price",      f"${price}" if price else "N/A")
c2.metric("Market Cap", fmt_money(info.get("marketCap")))
c3.metric("Sector",     info.get("sector", "N/A"))
c4.metric("Industry",   info.get("industry", "N/A"))

# --- Compute metrics ---
val  = compute_valuation(info)
grw  = compute_growth(info, inc)
prof = compute_profitability(info)
hlth = compute_health(info)
fut  = compute_future(info)
tech = compute_technicals(hist)

cat_scores = {
    "Valuation":      score_val(val),
    "Growth":         score_growth(grw),
    "Profitability":  score_profit(prof),
    "Future Outlook": score_future(fut),
    "Technicals":     score_tech(tech),
}
total, verd, color = get_verdict(cat_scores)

# --- Verdict banner ---
st.markdown("---")
v1, v2 = st.columns([1, 2])
with v1:
    st.markdown(f"### Verdict: :{color}[{verd}]")
    st.markdown(f"**Composite Score:** {total:+.2f}  (−1 undervalued → +1 overvalued)")
with v2:
    st.dataframe(pd.DataFrame({
        "Category": list(cat_scores.keys()),
        "Score":    [f"{v:+.2f}" for v in cat_scores.values()],
        "Weight":   [f"{WEIGHTS[c]:.0%}" for c in cat_scores],
    }), hide_index=True, use_container_width=True)
st.caption("Heuristic screen only — not investment advice.")

# --- Price chart ---
st.markdown("---")
st.subheader("Price Chart (2Y)")
if not hist.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color="#1f77b4")))
    if len(hist) >= 50:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),
                                 name="50-Day MA", line=dict(color="orange", dash="dot")))
    if len(hist) >= 200:
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(),
                                 name="200-Day MA", line=dict(color="red", dash="dot")))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------
st.markdown("---")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Valuation", "Growth", "Profitability & Health",
    "Future Outlook", "Technicals",
    "📐 DCF Calculator", "👥 Peer Comparison"
])

PCT_GRW  = ["Revenue Growth (YoY)", "Earnings Growth (YoY)",
            "Quarterly Earnings Growth (YoY)", "Revenue 3-4Y CAGR", "Net Income 3-4Y CAGR"]
PCT_PROF = ["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"]
PCT_FUT  = ["Implied Upside"]
PCT_TECH = ["% From 52W High", "% From 52W Low"]
MON_FUT  = ["Current Price", "Analyst Target Mean", "Analyst Target High", "Analyst Target Low"]
MON_TECH = ["Current Price", "50-Day MA", "200-Day MA", "52-Week High", "52-Week Low"]

with tab1: show_table(val)
with tab2: show_table(grw, pct_keys=PCT_GRW)
with tab3:
    st.markdown("**Profitability**")
    show_table(prof, pct_keys=PCT_PROF)
    st.markdown("**Financial Health**")
    show_table(hlth, money_keys=["Free Cash Flow"])
with tab4: show_table(fut, pct_keys=PCT_FUT, money_keys=MON_FUT)
with tab5: show_table(tech, pct_keys=PCT_TECH, money_keys=MON_TECH)

# -----------------------------------------------------------------------
# DCF Calculator
# -----------------------------------------------------------------------
with tab6:
    st.subheader("📐 DCF Intrinsic Value Calculator")
    st.caption("Adjust the sliders. The model projects free cash flows and discounts them back to today.")

    try:
        base_fcf = cf_stmt.loc["Free Cash Flow"].iloc[0] if cf_stmt is not None and not cf_stmt.empty else None
    except Exception:
        base_fcf = sg(info, "freeCashflow")
    try:
        base_rev = inc.loc["Total Revenue"].iloc[0] if inc is not None and not inc.empty else None
    except Exception:
        base_rev = sg(info, "totalRevenue")

    shares = sg(info, "sharesOutstanding", "impliedSharesOutstanding")
    try:
        total_debt = bal.loc["Total Debt"].iloc[0] if bal is not None and not bal.empty else 0
    except Exception:
        total_debt = 0
    try:
        cash_val = bal.loc["Cash And Cash Equivalents"].iloc[0] if bal is not None and not bal.empty else 0
    except Exception:
        cash_val = sg(info, "totalCash", default=0)

    net_debt  = (total_debt or 0) - (cash_val or 0)
    cur_price = sg(info, "currentPrice", "regularMarketPrice")

    st.markdown("**Company Financials (auto-filled from latest filings)**")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Latest FCF",     fmt_money(base_fcf))
    d2.metric("Latest Revenue", fmt_money(base_rev))
    d3.metric("Net Debt",       fmt_money(net_debt))
    d4.metric("Shares Out.",    fmt_money(shares))

    st.markdown("**Your Assumptions**")
    a1, a2, a3, a4 = st.columns(4)
    growth_rate     = a1.slider("FCF Growth Rate",      0, 40, 10, 1, format="%d%%", key="dcf_growth")     / 100
    terminal_growth = a2.slider("Terminal Growth Rate", 0,  5,  2, 1, format="%d%%", key="dcf_terminal")   / 100
    discount_rate   = a3.slider("Discount Rate (WACC)", 5, 20, 10, 1, format="%d%%", key="dcf_discount")   / 100
    proj_years      = a4.slider("Projection Years",     5, 15, 10,                   key="dcf_years")

    if discount_rate <= terminal_growth:
        st.error("Discount rate must be greater than terminal growth rate.")
    else:
        intrinsic, cf_rows, tv_pv = run_dcf(
            base_fcf, base_rev, growth_rate, terminal_growth,
            discount_rate, proj_years, shares, net_debt
        )
        if intrinsic is None:
            st.warning("Not enough data to run DCF — missing FCF or share count.")
        else:
            mos = (intrinsic - cur_price) / intrinsic if cur_price else None
            r1, r2, r3 = st.columns(3)
            r1.metric("Intrinsic Value / Share", f"${intrinsic:.2f}")
            r2.metric("Current Price", f"${cur_price:.2f}" if cur_price else "N/A")
            if mos is not None:
                lbl = "Upside" if mos > 0 else "Downside"
                tag = "Undervalued" if mos > 0 else "Overvalued"
                r3.metric(lbl, f"{mos * 100:.1f}%", delta=tag,
                          delta_color="normal" if mos > 0 else "inverse")

            st.markdown("**Present Value Breakdown**")
            df_cf = pd.DataFrame(cf_rows)
            fig_d = go.Figure(go.Bar(
                x=df_cf["Year"].tolist() + ["Terminal Value"],
                y=df_cf["PV ($M)"].tolist() + [tv_pv],
                marker_color=["#1f77b4"] * len(df_cf) + ["#ff7f0e"]
            ))
            fig_d.update_layout(title="PV of Projected FCFs + Terminal Value ($M)",
                                height=340, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_d, use_container_width=True)

            st.markdown("**Sensitivity Table — Intrinsic Value / Share**")
            g_range = [round(growth_rate + d, 2) for d in [-0.04, -0.02, 0, 0.02, 0.04]]
            d_range = [round(discount_rate + d, 3) for d in [-0.02, -0.01, 0, 0.01, 0.02]]
            d_valid = [d for d in d_range if d > terminal_growth and d > 0]
            sens = {}
            for dr in d_valid:
                col_vals = []
                for gr in g_range:
                    iv, _, _ = run_dcf(base_fcf, base_rev, gr, terminal_growth, dr, proj_years, shares, net_debt)
                    col_vals.append(f"${iv:.0f}" if iv else "N/A")
                sens[f"WACC {dr*100:.1f}%"] = col_vals
            sens_df = pd.DataFrame(sens, index=[f"Growth {g*100:.0f}%" for g in g_range])
            st.dataframe(sens_df, use_container_width=True)
            st.caption("Rows = FCF growth rate | Columns = discount rate (WACC)")

# -----------------------------------------------------------------------
# Peer Comparison
# -----------------------------------------------------------------------
with tab7:
    st.subheader("👥 Peer Comparison")

    DEFAULT_PEERS = {
        "AAPL": "MSFT, GOOGL, META, AMZN",
        "TSLA": "F, GM, RIVN, NIO",
        "MSFT": "AAPL, GOOGL, AMZN, CRM",
        "AMZN": "MSFT, GOOGL, BABA, WMT",
        "GOOGL": "MSFT, META, AMZN, AAPL",
        "META":  "GOOGL, SNAP, PINS, TWTR",
        "NVDA":  "AMD, INTC, QCOM, TSM",
        "JPM":   "BAC, GS, MS, WFC",
        "JNJ":   "PFE, MRK, ABT, UNH",
        "XOM":   "CVX, BP, SHEL, COP",
    }

    peer_input = st.text_input(
        "Enter peer tickers (comma-separated)",
        value=DEFAULT_PEERS.get(symbol, ""),
        key="peer_input",
        help="Type competitor ticker symbols, e.g. MSFT, GOOGL, META"
    )
    run_peers = st.button("Compare Peers", type="secondary", key="btn_peers")

    # Fetch peers and store in session_state so the comparison
    # survives subsequent widget interactions without re-fetching
    if run_peers and peer_input:
        peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()][:6]
        with st.spinner(f"Fetching data for {', '.join(peers)}..."):
            peer_infos = {symbol: info}
            for p in peers:
                peer_infos[p] = fetch_peer(p)
        st.session_state["peer_infos"] = peer_infos

    peer_infos = st.session_state.get("peer_infos")

    if peer_infos:
        # Comparison table
        rows = []
        for ticker, inf in peer_infos.items():
            if not inf: continue
            row = {
                "Ticker":  ticker,
                "Name":    (inf.get("longName") or ticker)[:22],
                "Mkt Cap": fmt_money(inf.get("marketCap")),
            }
            for label, (key, is_pct) in PEER_KEYS.items():
                v = sg(inf, key)
                row[label] = (fmt_pct(v) if is_pct else fmt_num(v)) if v is not None else "N/A"
            rows.append(row)

        comp_df = pd.DataFrame(rows).set_index("Ticker")
        st.dataframe(comp_df, use_container_width=True)

        # Bar charts
        st.markdown("**Visual Comparison**")
        chart_cols   = st.columns(2)
        chart_metrics = ["P/E", "EV/EBITDA", "Net Margin", "ROE"]
        for i, metric in enumerate(chart_metrics):
            key, is_pct = PEER_KEYS[metric]
            chart_data = []
            for ticker, inf in peer_infos.items():
                v = sg(inf, key)
                if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                    chart_data.append({"Ticker": ticker, "Value": v * 100 if is_pct else v})
            if chart_data:
                cdf = pd.DataFrame(chart_data).sort_values("Value")
                bar_colors = ["#e74c3c" if t == symbol else "#3498db" for t in cdf["Ticker"]]
                fig_p = go.Figure(go.Bar(x=cdf["Ticker"], y=cdf["Value"], marker_color=bar_colors))
                fig_p.update_layout(
                    title=f"{metric}{' (%)' if is_pct else ''}  — red = {symbol}",
                    height=280, showlegend=False,
                    margin=dict(l=10, r=10, t=40, b=10)
                )
                chart_cols[i % 2].plotly_chart(fig_p, use_container_width=True)

        # Percentile ranking
        st.markdown(f"**{symbol} Percentile Ranking vs Peers**")
        pct_rows = []
        for label, (key, is_pct) in PEER_KEYS.items():
            vals = {}
            for ticker, inf in peer_infos.items():
                v = sg(inf, key)
                if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                    vals[ticker] = v
            if symbol in vals and len(vals) > 1:
                main_val    = vals[symbol]
                sorted_vals = sorted(vals.values())
                rank        = sorted_vals.index(main_val) + 1
                n           = len(sorted_vals)
                pct_rank    = (rank / n * 100) if is_pct else ((n - rank + 1) / n * 100)
                interp = ("✅ Better than most peers" if pct_rank >= 60
                          else "⚠️ Middle of the pack" if pct_rank >= 40
                          else "🔴 Lags most peers")
                pct_rows.append({
                    "Metric":          label,
                    f"{symbol} Value": fmt_pct(main_val) if is_pct else fmt_num(main_val),
                    "Percentile":      f"{pct_rank:.0f}th",
                    "Interpretation":  interp,
                })
        if pct_rows:
            st.dataframe(pd.DataFrame(pct_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Enter peer tickers above and click **Compare Peers** to run the comparison.")
