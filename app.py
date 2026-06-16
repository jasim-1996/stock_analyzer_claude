"""
Stock Analyzer - Web App (Streamlit)

Run locally:
    pip install streamlit yfinance pandas numpy ta plotly
    streamlit run app.py

Deploy free:
    1. Push this file + requirements.txt to a GitHub repo
    2. Go to https://share.streamlit.io
    3. Connect your repo and deploy
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Data fetching (cached so repeated lookups are fast)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_ticker_data(symbol):
    t = yf.Ticker(symbol)
    info = t.info
    if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
        return None

    hist = t.history(period="2y")
    income_stmt = t.income_stmt
    quarterly_income = t.quarterly_income_stmt
    balance_sheet = t.balance_sheet
    cashflow = t.cashflow

    return {
        "info": info,
        "hist": hist,
        "income_stmt": income_stmt,
        "quarterly_income": quarterly_income,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_get(d, key, default=None):
    val = d.get(key, default)
    if val is None:
        return default
    return val


def fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x * 100:.2f}%"


def fmt_num(x, dec=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{dec}f}"


def fmt_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    abs_x = abs(x)
    if abs_x >= 1e12:
        return f"${x/1e12:.2f}T"
    if abs_x >= 1e9:
        return f"${x/1e9:.2f}B"
    if abs_x >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"


def cagr(series):
    series = series.dropna()
    if len(series) < 2:
        return None
    vals = series.values[::-1]
    start, end = vals[0], vals[-1]
    years = len(vals) - 1
    if start <= 0 or end <= 0 or years <= 0:
        return None
    return (end / start) ** (1 / years) - 1


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_valuation_metrics(info):
    return {
        "Trailing P/E": safe_get(info, "trailingPE"),
        "Forward P/E": safe_get(info, "forwardPE"),
        "PEG Ratio": safe_get(info, "pegRatio") or safe_get(info, "trailingPegRatio"),
        "Price/Book": safe_get(info, "priceToBook"),
        "Price/Sales": safe_get(info, "priceToSalesTrailing12Months"),
        "EV/EBITDA": safe_get(info, "enterpriseToEbitda"),
        "EV/Revenue": safe_get(info, "enterpriseToRevenue"),
    }


def compute_growth_metrics(info, income_stmt):
    metrics = {
        "Revenue Growth (YoY)": safe_get(info, "revenueGrowth"),
        "Earnings Growth (YoY)": safe_get(info, "earningsGrowth"),
        "Quarterly Earnings Growth (YoY)": safe_get(info, "earningsQuarterlyGrowth"),
    }
    try:
        rev = income_stmt.loc["Total Revenue"]
        metrics["Revenue 3-4Y CAGR"] = cagr(rev)
    except Exception:
        metrics["Revenue 3-4Y CAGR"] = None
    try:
        ni = income_stmt.loc["Net Income"]
        metrics["Net Income 3-4Y CAGR"] = cagr(ni)
    except Exception:
        metrics["Net Income 3-4Y CAGR"] = None
    return metrics


def compute_profitability_metrics(info):
    return {
        "Gross Margin": safe_get(info, "grossMargins"),
        "Operating Margin": safe_get(info, "operatingMargins"),
        "Net Margin": safe_get(info, "profitMargins"),
        "ROE": safe_get(info, "returnOnEquity"),
        "ROA": safe_get(info, "returnOnAssets"),
    }


def compute_financial_health_metrics(info):
    return {
        "Debt/Equity": safe_get(info, "debtToEquity"),
        "Current Ratio": safe_get(info, "currentRatio"),
        "Quick Ratio": safe_get(info, "quickRatio"),
        "Free Cash Flow": safe_get(info, "freeCashflow"),
    }


def compute_future_expectations(info):
    current_price = safe_get(info, "currentPrice") or safe_get(info, "regularMarketPrice")
    target_mean = safe_get(info, "targetMeanPrice")
    upside = None
    if current_price and target_mean:
        upside = (target_mean - current_price) / current_price
    return {
        "Current Price": current_price,
        "Analyst Target Mean": target_mean,
        "Analyst Target High": safe_get(info, "targetHighPrice"),
        "Analyst Target Low": safe_get(info, "targetLowPrice"),
        "Implied Upside": upside,
        "Recommendation": safe_get(info, "recommendationKey"),
        "Number of Analyst Opinions": safe_get(info, "numberOfAnalystOpinions"),
        "Forward EPS": safe_get(info, "forwardEps"),
        "Trailing EPS": safe_get(info, "trailingEps"),
    }


def compute_technical_metrics(hist):
    if hist.empty:
        return {}
    close = hist["Close"]
    current = close.iloc[-1]

    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None

    rsi_series = ta.momentum.RSIIndicator(close, window=14).rsi()
    rsi = rsi_series.iloc[-1] if not rsi_series.empty else None

    macd_obj = ta.trend.MACD(close)
    macd_line = macd_obj.macd().iloc[-1]
    macd_signal = macd_obj.macd_signal().iloc[-1]

    high_52w = close[-252:].max()
    low_52w = close[-252:].min()

    return {
        "Current Price": current,
        "50-Day MA": ma50,
        "200-Day MA": ma200,
        "RSI (14)": rsi,
        "MACD": macd_line,
        "MACD Signal": macd_signal,
        "52-Week High": high_52w,
        "52-Week Low": low_52w,
        "% From 52W High": (current - high_52w) / high_52w,
        "% From 52W Low": (current - low_52w) / low_52w,
    }


# ---------------------------------------------------------------------------
# Scoring system (-1 = undervalued/bullish, +1 = overvalued/bearish)
# ---------------------------------------------------------------------------

def score_valuation(metrics):
    scores = []
    pe = metrics.get("Trailing P/E")
    if pe:
        scores.append(-1 if pe < 15 else (1 if pe > 30 else 0))
    peg = metrics.get("PEG Ratio")
    if peg:
        scores.append(-1 if peg < 1 else (1 if peg > 2 else 0))
    pb = metrics.get("Price/Book")
    if pb:
        scores.append(-1 if pb < 1 else (1 if pb > 5 else 0))
    ev_ebitda = metrics.get("EV/EBITDA")
    if ev_ebitda:
        scores.append(-1 if ev_ebitda < 8 else (1 if ev_ebitda > 18 else 0))
    return np.mean(scores) if scores else 0


def score_growth(metrics):
    scores = []
    for key, hi, lo in [
        ("Revenue Growth (YoY)", 0.15, 0),
        ("Earnings Growth (YoY)", 0.15, 0),
        ("Revenue 3-4Y CAGR", 0.10, 0),
    ]:
        v = metrics.get(key)
        if v is not None:
            scores.append(-1 if v > hi else (1 if v < lo else 0))
    return np.mean(scores) if scores else 0


def score_profitability(metrics):
    scores = []
    nm = metrics.get("Net Margin")
    if nm is not None:
        scores.append(-1 if nm > 0.15 else (1 if nm < 0.05 else 0))
    roe = metrics.get("ROE")
    if roe is not None:
        scores.append(-1 if roe > 0.15 else (1 if roe < 0.05 else 0))
    return np.mean(scores) if scores else 0


def score_future(metrics):
    scores = []
    upside = metrics.get("Implied Upside")
    if upside is not None:
        scores.append(-1 if upside > 0.10 else (1 if upside < -0.05 else 0))
    rec = metrics.get("Recommendation")
    if rec:
        if rec in ("strong_buy", "buy"):
            scores.append(-1)
        elif rec in ("sell", "strong_sell"):
            scores.append(1)
        else:
            scores.append(0)
    return np.mean(scores) if scores else 0


def score_technicals(metrics):
    scores = []
    rsi = metrics.get("RSI (14)")
    if rsi is not None:
        scores.append(-1 if rsi < 30 else (1 if rsi > 70 else 0))
    price = metrics.get("Current Price")
    ma50 = metrics.get("50-Day MA")
    ma200 = metrics.get("200-Day MA")
    if price and ma50:
        scores.append(-0.5 if price < ma50 else 0.5)
    if price and ma200:
        scores.append(-0.5 if price < ma200 else 0.5)
    macd, macd_sig = metrics.get("MACD"), metrics.get("MACD Signal")
    if macd is not None and macd_sig is not None:
        scores.append(-0.5 if macd > macd_sig else 0.5)
    return np.mean(scores) if scores else 0


WEIGHTS = {
    "Valuation": 0.35,
    "Growth": 0.20,
    "Profitability": 0.15,
    "Future Outlook": 0.15,
    "Technicals": 0.15,
}


def composite_verdict(category_scores):
    total = sum(category_scores[c] * WEIGHTS[c] for c in WEIGHTS)
    if total <= -0.3:
        verdict = "UNDERVALUED"
        color = "green"
    elif total >= 0.3:
        verdict = "OVERVALUED"
        color = "red"
    else:
        verdict = "FAIRLY VALUED"
        color = "orange"
    return total, verdict, color


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock Analyzer", page_icon="📊", layout="wide")

st.title("📊 Stock Analyzer")
st.caption("Enter a ticker symbol to get a valuation verdict based on fundamentals, growth, profitability, analyst outlook, and technicals.")

symbol = st.text_input("Ticker Symbol", value="AAPL").strip().upper()
go_button = st.button("Analyze", type="primary")

if go_button and symbol:
    with st.spinner(f"Fetching data for {symbol}..."):
        data = get_ticker_data(symbol)

    if data is None:
        st.error(f"No data found for '{symbol}'. Check the ticker symbol and try again.")
    else:
        info = data["info"]

        # --- Header ---
        st.header(f"{info.get('longName', symbol)} ({symbol})")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sector", info.get("sector", "N/A"))
        col2.metric("Industry", info.get("industry", "N/A"))
        col3.metric("Market Cap", fmt_money(info.get("marketCap")))

        # --- Compute everything ---
        valuation = compute_valuation_metrics(info)
        growth = compute_growth_metrics(info, data["income_stmt"])
        profitability = compute_profitability_metrics(info)
        health = compute_financial_health_metrics(info)
        future = compute_future_expectations(info)
        technicals = compute_technical_metrics(data["hist"])

        cat_scores = {
            "Valuation": score_valuation(valuation),
            "Growth": score_growth(growth),
            "Profitability": score_profitability(profitability),
            "Future Outlook": score_future(future),
            "Technicals": score_technicals(technicals),
        }
        total_score, verdict, color = composite_verdict(cat_scores)

        # --- Verdict banner ---
        st.markdown("---")
        v1, v2 = st.columns([1, 2])
        with v1:
            st.markdown(f"### Verdict: :{color}[{verdict}]")
            st.markdown(f"**Composite Score:** {total_score:+.2f}  (range -1 to +1)")
        with v2:
            score_df = pd.DataFrame({
                "Category": list(cat_scores.keys()),
                "Score": list(cat_scores.values()),
                "Weight": [f"{WEIGHTS[c]:.0%}" for c in cat_scores],
            })
            st.dataframe(score_df, hide_index=True, use_container_width=True)

        st.caption("Score convention: -1 = undervalued/bullish signal, +1 = overvalued/bearish signal. This is a heuristic screen, not investment advice.")

        # --- Price chart ---
        st.markdown("---")
        st.subheader("Price Chart (2Y)")
        hist = data["hist"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color="#1f77b4")))
        if len(hist) >= 50:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="50-Day MA", line=dict(color="orange", dash="dot")))
        if len(hist) >= 200:
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(), name="200-Day MA", line=dict(color="red", dash="dot")))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # --- Detail sections ---
        st.markdown("---")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Valuation", "Growth", "Profitability & Health", "Future Outlook", "Technicals"])

        def show_table(metrics, pct_keys=None, money_keys=None):
            pct_keys = pct_keys or []
            money_keys = money_keys or []
            rows = []
            for k, v in metrics.items():
                if k in pct_keys:
                    display = fmt_pct(v)
                elif k in money_keys:
                    display = fmt_money(v)
                elif isinstance(v, float):
                    display = fmt_num(v)
                else:
                    display = v if v is not None else "N/A"
                rows.append({"Metric": k, "Value": display})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        with tab1:
            show_table(valuation)

        with tab2:
            show_table(growth, pct_keys=["Revenue Growth (YoY)", "Earnings Growth (YoY)",
                                          "Quarterly Earnings Growth (YoY)", "Revenue 3-4Y CAGR", "Net Income 3-4Y CAGR"])

        with tab3:
            st.markdown("**Profitability**")
            show_table(profitability, pct_keys=["Gross Margin", "Operating Margin", "Net Margin", "ROE", "ROA"])
            st.markdown("**Financial Health**")
            show_table(health, money_keys=["Free Cash Flow"])

        with tab4:
            show_table(future, pct_keys=["Implied Upside"], money_keys=["Current Price", "Analyst Target Mean", "Analyst Target High", "Analyst Target Low"])

        with tab5:
            show_table(technicals, pct_keys=["% From 52W High", "% From 52W Low"], money_keys=["Current Price", "50-Day MA", "200-Day MA", "52-Week High", "52-Week Low"])

else:
    st.info("Enter a ticker symbol and click Analyze to get started.")
