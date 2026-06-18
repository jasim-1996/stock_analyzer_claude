"""
Stock Analyzer — Equity Terminal
Uses yfinance for data. Works locally out of the box.

Run locally:
    pip install streamlit yfinance pandas numpy ta plotly requests
    streamlit run app.py

Deploy on Streamlit Cloud:
    Push app.py + requirements.txt to GitHub → share.streamlit.io
    If you hit Yahoo Finance rate limits on the cloud, add an Upstash
    Redis URL to Streamlit Secrets:  REDIS_URL = "rediss://..."
"""

import json, time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import ta
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Redis cache (optional) — persists across Streamlit Cloud restarts.
# Falls back silently when not configured (local dev works without it).
# ---------------------------------------------------------------------------

def get_redis():
    try:
        import redis
        url = st.secrets.get("REDIS_URL", "")
        if not url:
            return None
        r = redis.from_url(url, decode_responses=True, socket_timeout=5)
        r.ping()
        return r
    except Exception:
        return None

REDIS      = get_redis()
CACHE_TTL  = 86400  # 24 hours

def redis_get(key):
    if REDIS:
        try:
            val = REDIS.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None
    return None

def redis_set(key, value):
    if REDIS:
        try:
            REDIS.setex(key, CACHE_TTL, json.dumps(value, default=str))
        except Exception:
            pass

# ---------------------------------------------------------------------------
# yfinance fetch — Redis cache → session_state → live fetch
# ---------------------------------------------------------------------------

def yf_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.5",
    })
    return s

def df_to_json(df):
    if df is None or (hasattr(df, "empty") and df.empty):
        return None
    return df.reset_index().to_dict(orient="list")

def json_to_df(d):
    if d is None:
        return pd.DataFrame()
    df = pd.DataFrame(d)
    for col in ("index", "Date"):
        if col in df.columns:
            df = df.set_index(col)
            if col == "Date":
                df.index = pd.to_datetime(df.index)
            break
    return df

def live_fetch(symbol):
    t    = yf.Ticker(symbol, session=yf_session())
    info = t.info
    if not info or (info.get("regularMarketPrice") is None
                    and info.get("currentPrice") is None):
        raise ValueError(f"No data found for '{symbol}'.")
    return {
        "info":          info,
        "hist":          df_to_json(t.history(period="2y")),
        "income_stmt":   df_to_json(t.income_stmt),
        "balance_sheet": df_to_json(t.balance_sheet),
        "cashflow":      df_to_json(t.cashflow),
    }

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_all(symbol):
    try:
        key    = f"yf:{symbol}"
        cached = redis_get(key)
        if not cached:
            cached = live_fetch(symbol)
            redis_set(key, cached)
        return {
            "info":          cached["info"],
            "hist":          json_to_df(cached["hist"]),
            "income_stmt":   json_to_df(cached["income_stmt"]),
            "balance_sheet": json_to_df(cached["balance_sheet"]),
            "cashflow":      json_to_df(cached["cashflow"]),
        }
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer(ticker):
    try:
        key    = f"yf:{ticker}"
        cached = redis_get(key)
        if cached:
            return cached["info"]
        info = yf.Ticker(ticker, session=yf_session()).info
        redis_set(key, {"info": info, "hist": None, "income_stmt": None,
                        "balance_sheet": None, "cashflow": None})
        return info or {}
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

def cagr(series):
    series = series.dropna()
    if len(series) < 2: return None
    vals = series.values[::-1]
    s, e = vals[0], vals[-1]
    n = len(vals) - 1
    if s <= 0 or e <= 0 or n <= 0: return None
    return (e / s) ** (1 / n) - 1

def show_metric_table(rows):
    html = '<table class="metric-table">'
    for label, value in rows:
        if value == "":
            html += f'<tr class="section-header"><td colspan="2">{label}</td></tr>'
        else:
            html += (f'<tr><td class="metric-label">{label}</td>'
                     f'<td class="metric-value">{value}</td></tr>')
    html += "</table>"
    st.markdown(html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_valuation(info):
    return [
        ("Trailing P/E",  fmt_num(sg(info, "trailingPE"))),
        ("Forward P/E",   fmt_num(sg(info, "forwardPE"))),
        ("PEG Ratio",     fmt_num(sg(info, "pegRatio", "trailingPegRatio"))),
        ("Price / Book",  fmt_num(sg(info, "priceToBook"))),
        ("Price / Sales", fmt_num(sg(info, "priceToSalesTrailing12Months"))),
        ("EV / EBITDA",   fmt_num(sg(info, "enterpriseToEbitda"))),
        ("EV / Revenue",  fmt_num(sg(info, "enterpriseToRevenue"))),
        ("Div Yield",     fmt_pct(sg(info, "dividendYield"))),
    ]

def compute_growth(info, inc):
    rows = [
        ("Revenue Growth (YoY)",            fmt_pct(sg(info, "revenueGrowth"))),
        ("Earnings Growth (YoY)",           fmt_pct(sg(info, "earningsGrowth"))),
        ("Qtrly Earnings Growth (YoY)",     fmt_pct(sg(info, "earningsQuarterlyGrowth"))),
        ("EPS (TTM)",                       fmt_num(sg(info, "trailingEps"))),
        ("Forward EPS",                     fmt_num(sg(info, "forwardEps"))),
    ]
    try:
        rev = inc.loc["Total Revenue"]
        rows.append(("Revenue 3-4Y CAGR", fmt_pct(cagr(rev))))
    except Exception: pass
    try:
        ni = inc.loc["Net Income"]
        rows.append(("Net Income 3-4Y CAGR", fmt_pct(cagr(ni))))
    except Exception: pass
    return rows

def compute_profitability(info):
    return [
        ("Gross Margin",     fmt_pct(sg(info, "grossMargins"))),
        ("Operating Margin", fmt_pct(sg(info, "operatingMargins"))),
        ("Net Margin",       fmt_pct(sg(info, "profitMargins"))),
        ("ROE",              fmt_pct(sg(info, "returnOnEquity"))),
        ("ROA",              fmt_pct(sg(info, "returnOnAssets"))),
    ]

def compute_health(info):
    return [
        ("Debt / Equity",  fmt_num(sg(info, "debtToEquity"))),
        ("Current Ratio",  fmt_num(sg(info, "currentRatio"))),
        ("Quick Ratio",    fmt_num(sg(info, "quickRatio"))),
        ("Free Cash Flow", fmt_money(sg(info, "freeCashflow"))),
        ("Total Cash",     fmt_money(sg(info, "totalCash"))),
        ("Total Debt",     fmt_money(sg(info, "totalDebt"))),
    ]

def compute_future(info):
    price  = sg(info, "currentPrice", "regularMarketPrice")
    target = sg(info, "targetMeanPrice")
    upside = (target - price) / price if price and target else None
    return [
        ("Current Price",       f"${price:.2f}" if price else "N/A"),
        ("Analyst Target Mean", f"${target:.2f}" if target else "N/A"),
        ("Analyst Target High", f"${sg(info, 'targetHighPrice'):.2f}" if sg(info, "targetHighPrice") else "N/A"),
        ("Analyst Target Low",  f"${sg(info, 'targetLowPrice'):.2f}"  if sg(info, "targetLowPrice")  else "N/A"),
        ("Implied Upside",      fmt_pct(upside)),
        ("Recommendation",      str(sg(info, "recommendationKey", default="N/A")).upper()),
        ("# Analyst Opinions",  str(sg(info, "numberOfAnalystOpinions", default="N/A"))),
        ("Forward EPS",         fmt_num(sg(info, "forwardEps"))),
        ("Trailing EPS",        fmt_num(sg(info, "trailingEps"))),
    ]

def compute_technicals(hist):
    if hist is None or hist.empty: return []
    close   = hist["Close"]
    current = close.iloc[-1]
    ma50    = close.rolling(50).mean().iloc[-1]  if len(close) >= 50  else None
    ma200   = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else None
    rsi     = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1] if len(close) > 14 else None
    macd_o  = ta.trend.MACD(close)
    macd    = macd_o.macd().iloc[-1]
    msig    = macd_o.macd_signal().iloc[-1]
    bb      = ta.volatility.BollingerBands(close)
    h52, l52 = close[-252:].max(), close[-252:].min()
    return [
        ("Current Price",       f"${current:.2f}"),
        ("50-Day MA",           f"${ma50:.2f}" if ma50 else "N/A"),
        ("200-Day MA",          f"${ma200:.2f}" if ma200 else "N/A"),
        ("Price vs 50-Day MA",  "Above ✅" if ma50 and current > ma50 else "Below ⚠️"),
        ("Price vs 200-Day MA", "Above ✅" if ma200 and current > ma200 else "Below ⚠️"),
        ("RSI (14)",            fmt_num(rsi)),
        ("RSI Signal",          "Oversold 🟢" if rsi and rsi < 30 else ("Overbought 🔴" if rsi and rsi > 70 else "Neutral")),
        ("MACD",                fmt_num(macd)),
        ("MACD Signal Line",    fmt_num(msig)),
        ("MACD Cross",          "Bullish ✅" if macd and msig and macd > msig else "Bearish ⚠️"),
        ("Bollinger Upper",     f"${bb.bollinger_hband().iloc[-1]:.2f}"),
        ("Bollinger Lower",     f"${bb.bollinger_lband().iloc[-1]:.2f}"),
        ("52-Week High",        f"${h52:.2f}"),
        ("52-Week Low",         f"${l52:.2f}"),
        ("% From 52W High",     fmt_pct((current - h52) / h52)),
        ("% From 52W Low",      fmt_pct((current - l52) / l52)),
    ]

# ---------------------------------------------------------------------------
# Scoring  (-1 = undervalued/bullish · +1 = overvalued/bearish)
# ---------------------------------------------------------------------------

def score_val(info):
    s = []
    pe = sg(info, "trailingPE")
    if pe:  s.append(-1 if pe < 15 else (1 if pe > 30 else 0))
    peg = sg(info, "pegRatio", "trailingPegRatio")
    if peg: s.append(-1 if peg < 1 else (1 if peg > 2 else 0))
    pb = sg(info, "priceToBook")
    if pb:  s.append(-1 if pb < 1 else (1 if pb > 5 else 0))
    ev = sg(info, "enterpriseToEbitda")
    if ev:  s.append(-1 if ev < 8 else (1 if ev > 18 else 0))
    return np.mean(s) if s else 0

def score_growth(info, inc):
    s = []
    for k, hi, lo in [("revenueGrowth", 0.15, 0),
                       ("earningsGrowth", 0.15, 0)]:
        v = sg(info, k)
        if v is not None: s.append(-1 if v > hi else (1 if v < lo else 0))
    try:
        rev = inc.loc["Total Revenue"]
        c = cagr(rev)
        if c is not None: s.append(-1 if c > 0.10 else (1 if c < 0 else 0))
    except Exception: pass
    return np.mean(s) if s else 0

def score_profit(info):
    s = []
    nm = sg(info, "profitMargins")
    if nm is not None: s.append(-1 if nm > 0.15 else (1 if nm < 0.05 else 0))
    roe = sg(info, "returnOnEquity")
    if roe is not None: s.append(-1 if roe > 0.15 else (1 if roe < 0.05 else 0))
    return np.mean(s) if s else 0

def score_future(info):
    s = []
    price  = sg(info, "currentPrice", "regularMarketPrice")
    target = sg(info, "targetMeanPrice")
    if price and target:
        up = (target - price) / price
        s.append(-1 if up > 0.10 else (1 if up < -0.05 else 0))
    rec = str(sg(info, "recommendationKey", default="")).lower()
    if rec in ("strong_buy", "buy"):    s.append(-1)
    elif rec in ("sell", "strong_sell"): s.append(1)
    elif rec: s.append(0)
    return np.mean(s) if s else 0

def score_tech(hist):
    if hist is None or hist.empty: return 0
    close = hist["Close"]
    s = []
    rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1] if len(close) > 14 else None
    if rsi is not None: s.append(-1 if rsi < 30 else (1 if rsi > 70 else 0))
    cur = close.iloc[-1]
    if len(close) >= 50:
        ma50 = close.rolling(50).mean().iloc[-1]
        s.append(-0.5 if cur < ma50 else 0.5)
    if len(close) >= 200:
        ma200 = close.rolling(200).mean().iloc[-1]
        s.append(-0.5 if cur < ma200 else 0.5)
    macd_o = ta.trend.MACD(close)
    macd   = macd_o.macd().iloc[-1]
    msig   = macd_o.macd_signal().iloc[-1]
    if macd is not None and msig is not None:
        s.append(-0.5 if macd > msig else 0.5)
    return np.mean(s) if s else 0

WEIGHTS = {"Valuation": 0.35, "Growth": 0.20, "Profitability": 0.15,
           "Future Outlook": 0.15, "Technicals": 0.15}

def get_verdict(scores):
    total = sum(scores[c] * WEIGHTS[c] for c in WEIGHTS)
    if   total <= -0.3: return total, "UNDERVALUED",   "#00e676", "green"
    elif total >=  0.3: return total, "OVERVALUED",    "#ff4444", "red"
    else:               return total, "FAIRLY VALUED",  "#f5a623", "orange"

# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------

def run_dcf(fcf, revenue, growth_rate, terminal_growth, discount_rate, years, shares, net_debt):
    base = fcf if (fcf and fcf > 0) else (revenue * 0.05 if revenue else None)
    if not base or not shares or shares == 0: return None, None, None
    cf_rows = []
    for y in range(1, years + 1):
        cf = base * ((1 + growth_rate) ** y)
        pv = cf / ((1 + discount_rate) ** y)
        cf_rows.append({"Year": f"Y{y}", "FCF ($M)": cf/1e6, "PV ($M)": pv/1e6})
    t_fcf = base * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    tv    = t_fcf / (discount_rate - terminal_growth)
    tv_pv = tv / ((1 + discount_rate) ** years)
    equity = (sum(r["PV ($M)"] for r in cf_rows) * 1e6 + tv_pv) - (net_debt or 0)
    return equity / shares, cf_rows, tv_pv / 1e6

# ---------------------------------------------------------------------------
# Custom CSS — Terminal dark theme (IBM Plex Mono / Sans)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --surface2:  #1a1e28;
    --border:    #252a38;
    --amber:     #f5a623;
    --green:     #00e676;
    --red:       #ff4444;
    --text:      #d4dae8;
    --muted:     #5a6278;
    --mono:      'IBM Plex Mono', monospace;
    --sans:      'IBM Plex Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"]     { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

/* Hero */
.hero { border-left: 3px solid var(--amber); padding: .5rem 0 .5rem 1.2rem; margin-bottom: 1.2rem; }
.hero-ticker { font-family: var(--mono); font-size: .8rem; font-weight: 600; letter-spacing: .18em; color: var(--amber); text-transform: uppercase; margin-bottom: .15rem; }
.hero-name   { font-family: var(--sans); font-size: 1.9rem; font-weight: 600; color: #fff; line-height: 1.1; margin-bottom: .2rem; }
.hero-meta   { font-family: var(--mono); font-size: .72rem; color: var(--muted); letter-spacing: .04em; }

/* Stat cards */
.stat-row  { display: flex; gap: .9rem; margin: 1rem 0; flex-wrap: wrap; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: .8rem 1rem; flex: 1; min-width: 120px; }
.stat-label { font-family: var(--mono); font-size: .6rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-bottom: .25rem; }
.stat-value { font-family: var(--mono); font-size: 1.2rem; font-weight: 600; color: #fff; }
.stat-value.up   { color: var(--green); }
.stat-value.down { color: var(--red);   }

/* Verdict */
.verdict-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.3rem 1.6rem; margin: 1rem 0; display: flex; align-items: flex-start; gap: 2rem; flex-wrap: wrap; }
.verdict-label { font-family: var(--mono); font-size: .62rem; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; margin-bottom: .2rem; }
.verdict-text  { font-family: var(--mono); font-size: 1.9rem; font-weight: 600; letter-spacing: .05em; }
.verdict-text.green  { color: var(--green); }
.verdict-text.red    { color: var(--red);   }
.verdict-text.orange { color: var(--amber); }
.score-bar-wrap { flex: 1; min-width: 220px; }
.score-track { background: var(--border); border-radius: 2px; height: 5px; margin-top: .5rem; position: relative; }
.score-fill  { height: 100%; border-radius: 2px; }
.score-labels { display: flex; justify-content: space-between; font-family: var(--mono); font-size: .58rem; color: var(--muted); margin-top: .2rem; }
.cat-scores { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .75rem; }
.cat-pill   { font-family: var(--mono); font-size: .65rem; padding: .22rem .6rem; border-radius: 3px; background: var(--surface2); border: 1px solid var(--border); color: var(--text); }
.cat-pill.bull { border-color: var(--green); color: var(--green); }
.cat-pill.bear { border-color: var(--red);   color: var(--red);   }

/* Metric table */
.metric-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: .79rem; margin-top: .3rem; }
.metric-table tr { border-bottom: 1px solid var(--border); }
.metric-table tr:last-child { border-bottom: none; }
.metric-table td  { padding: .52rem .5rem; }
.metric-label { color: var(--muted); width: 55%; }
.metric-value { color: var(--text); text-align: right; font-weight: 500; }
.section-header td { font-size: .62rem; letter-spacing: .12em; text-transform: uppercase; color: var(--amber); padding-top: 1rem !important; font-weight: 600; }

/* Tabs */
[data-testid="stTabs"] button { font-family: var(--mono) !important; font-size: .7rem !important; letter-spacing: .08em !important; color: var(--muted) !important; text-transform: uppercase !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--amber) !important; border-bottom-color: var(--amber) !important; }

/* Input */
[data-testid="stTextInput"] input { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; font-family: var(--mono) !important; font-size: .9rem !important; border-radius: 5px !important; }
[data-testid="stTextInput"] input:focus { border-color: var(--amber) !important; box-shadow: 0 0 0 2px rgba(245,166,35,.15) !important; }

/* Buttons */
[data-testid="stBaseButton-primary"] > button  { background: var(--amber) !important; color: #0d0f14 !important; font-family: var(--mono) !important; font-weight: 600 !important; border: none !important; border-radius: 5px !important; }
[data-testid="stBaseButton-secondary"] > button { background: transparent !important; color: var(--amber) !important; font-family: var(--mono) !important; border: 1px solid var(--amber) !important; border-radius: 5px !important; }

/* Sliders */
[data-testid="stSlider"] label { font-family: var(--mono) !important; font-size: .7rem !important; color: var(--muted) !important; letter-spacing: .06em !important; }

/* Metrics */
[data-testid="stMetric"]      { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; padding: .75rem 1rem !important; }
[data-testid="stMetricLabel"] { font-family: var(--mono) !important; font-size: .62rem !important; letter-spacing: .1em !important; color: var(--muted) !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: var(--mono) !important; font-size: 1.05rem !important; font-weight: 600 !important; color: #fff !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 6px !important; }

/* Sidebar */
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] [data-testid="stMetric"] { background: var(--surface2) !important; }

/* DCF result cards */
.dcf-card  { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.1rem 1.4rem; margin: .8rem 0; text-align: center; }
.dcf-label { font-family: var(--mono); font-size: .62rem; letter-spacing: .12em; color: var(--muted); text-transform: uppercase; margin-bottom: .3rem; }
.dcf-value { font-family: var(--mono); font-size: 1.55rem; font-weight: 600; color: #fff; }
.dcf-delta { font-family: var(--mono); font-size: .8rem; margin-top: .2rem; }
.dcf-delta.up   { color: var(--green); }
.dcf-delta.down { color: var(--red); }

/* Section eyebrow */
.eyebrow { font-family: var(--mono); font-size: .62rem; letter-spacing: .16em; color: var(--amber); text-transform: uppercase; margin-bottom: .6rem; font-weight: 600; }

hr.divider { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }
</style>
"""

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Equity Terminal", page_icon="▸", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="eyebrow">Session</div>', unsafe_allow_html=True)
    st.caption("Results cached for 1 hour. No API key needed.")
    if st.button("Clear Cache", help="Force re-fetch all data"):
        for k in list(st.session_state.keys()):
            if k.startswith("yf_") or k in ("data","last_symbol","peer_infos"):
                del st.session_state[k]
        st.cache_data.clear()
        st.rerun()
    st.markdown("---")
    st.markdown('<div class="eyebrow">Examples</div>', unsafe_allow_html=True)
    st.caption("AAPL · MSFT · GOOGL · TSLA · NVDA\nJPM · AMZN · META · XOM · JNJ")

# Search bar
c1, c2 = st.columns([4, 1])
with c1:
    symbol = st.text_input("ticker", value=st.session_state.get("last_symbol", "AAPL"),
                           placeholder="Enter ticker symbol  e.g. AAPL",
                           label_visibility="collapsed").strip().upper()
with c2:
    analyze = st.button("Analyze ▸", type="primary", use_container_width=True)

if analyze and symbol:
    with st.spinner(f"Loading {symbol}..."):
        data = fetch_all(symbol)
    if "error" in data:
        st.error(data["error"])
        st.stop()
    st.session_state["data"]        = data
    st.session_state["last_symbol"] = symbol
    st.session_state["peer_infos"]  = None

if "data" not in st.session_state:
    st.markdown("""
<div style="padding:4rem 1rem;text-align:center;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.25em;color:#5a6278;text-transform:uppercase;margin-bottom:.8rem;">Equity Terminal</div>
  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:2rem;font-weight:600;color:#d4dae8;margin-bottom:.5rem;">Research any public stock.</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:#5a6278;">Valuation · Growth · DCF · Peer Comparison · Technicals</div>
</div>""", unsafe_allow_html=True)
    st.stop()

data    = st.session_state["data"]
symbol  = st.session_state["last_symbol"]
info    = data["info"]
hist    = data["hist"]
inc     = data["income_stmt"]
bal     = data["balance_sheet"]
cf      = data["cashflow"]

# ── Hero ──
price = sg(info, "currentPrice", "regularMarketPrice")
prev  = sg(info, "previousClose")
chg   = (price - prev) if price and prev else None
chgp  = chg / prev * 100 if chg and prev else None
css_c = "up" if chg and chg >= 0 else "down"
sign  = "+" if chg and chg >= 0 else ""

st.markdown(f"""
<div class="hero">
  <div class="hero-ticker">▸ {symbol}</div>
  <div class="hero-name">{info.get('longName', symbol)}</div>
  <div class="hero-meta">{info.get('sector','—')} &nbsp;·&nbsp; {info.get('industry','—')} &nbsp;·&nbsp; {info.get('exchange','—')}</div>
</div>""", unsafe_allow_html=True)

# ── Stat cards ──
mktcap = fmt_money(info.get("marketCap"))
h52    = sg(info, "fiftyTwoWeekHigh")
l52    = sg(info, "fiftyTwoWeekLow")
dy     = sg(info, "dividendYield")
beta   = sg(info, "beta")

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-label">Price</div>
    <div class="stat-value">${price:.2f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Day Change</div>
    <div class="stat-value {css_c}">{sign}{chg:.2f} ({sign}{chgp:.2f}%)</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Market Cap</div>
    <div class="stat-value">{mktcap}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">52W Range</div>
    <div class="stat-value">${l52:.0f} – ${h52:.0f}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Div Yield</div>
    <div class="stat-value">{f"{dy:.2f}%" if dy else "—"}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Beta</div>
    <div class="stat-value">{f"{beta:.2f}" if beta else "—"}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Scores & Verdict ──
cat_scores = {
    "Valuation":      score_val(info),
    "Growth":         score_growth(info, inc),
    "Profitability":  score_profit(info),
    "Future Outlook": score_future(info),
    "Technicals":     score_tech(hist),
}
total, verd, verd_hex, verd_class = get_verdict(cat_scores)
bar_pct = int((total + 1) / 2 * 100)

pills = ""
for cat, sc in cat_scores.items():
    cls  = "bull" if sc < -0.15 else ("bear" if sc > 0.15 else "")
    sign2 = "+" if sc <= 0 else ""
    pills += f'<span class="cat-pill {cls}">{cat}: {sign2}{sc:.2f}</span>'

st.markdown(f"""
<div class="verdict-wrap">
  <div>
    <div class="verdict-label">Overall Verdict</div>
    <div class="verdict-text {verd_class}">{verd}</div>
  </div>
  <div class="score-bar-wrap">
    <div class="verdict-label">Composite Score &nbsp;<strong style="color:#d4dae8;">{total:+.2f}</strong></div>
    <div class="score-track">
      <div class="score-fill" style="width:{bar_pct}%;background:{verd_hex};"></div>
    </div>
    <div class="score-labels"><span>Undervalued</span><span>Fair</span><span>Overvalued</span></div>
    <div class="cat-scores">{pills}</div>
  </div>
</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#5a6278;margin-bottom:1rem;">
  Heuristic screen only — not investment advice. Score: −1.0 (undervalued) to +1.0 (overvalued).
</div>""", unsafe_allow_html=True)

# ── Price Chart ──
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Price Chart — 2 Years</div>', unsafe_allow_html=True)

if hist is not None and not hist.empty:
    close = hist["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=close, name="Price",
        fill="tozeroy", line=dict(color="#f5a623", width=1.5),
        fillcolor="rgba(245,166,35,0.06)"
    ))
    if len(close) >= 50:
        fig.add_trace(go.Scatter(
            x=hist.index, y=close.rolling(50).mean(),
            name="50-Day MA", line=dict(color="#00e676", width=1, dash="dot")
        ))
    if len(close) >= 200:
        fig.add_trace(go.Scatter(
            x=hist.index, y=close.rolling(200).mean(),
            name="200-Day MA", line=dict(color="#ff4444", width=1, dash="dot")
        ))
    fig.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color="#5a6278", size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10, color="#5a6278"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, color="#5a6278"),
        yaxis=dict(showgrid=True, gridcolor="#1a1e28", tickformat="$,.0f", color="#5a6278"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── Tabs ──
st.markdown('<hr class="divider">', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "VALUATION", "GROWTH", "PROFITABILITY",
    "OUTLOOK", "TECHNICALS", "DCF MODEL", "PEER COMPARE"
])

with tab1: show_metric_table(compute_valuation(info))
with tab2: show_metric_table(compute_growth(info, inc))
with tab3:
    show_metric_table(
        [("PROFITABILITY", "")] + compute_profitability(info) +
        [("FINANCIAL HEALTH", "")] + compute_health(info)
    )
with tab4: show_metric_table(compute_future(info))
with tab5: show_metric_table(compute_technicals(hist))

# ── DCF ──
with tab6:
    st.markdown('<div class="eyebrow">DCF Intrinsic Value Model</div>', unsafe_allow_html=True)
    st.caption("Adjust assumptions below. Model discounts projected FCFs + terminal value to today.")

    try:
        base_fcf = cf.loc["Free Cash Flow"].iloc[0] if cf is not None and not cf.empty else None
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
    net_debt = (total_debt or 0) - (cash_val or 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest FCF",     fmt_money(base_fcf))
    m2.metric("Latest Revenue", fmt_money(base_rev))
    m3.metric("Net Debt",       fmt_money(net_debt))
    m4.metric("Shares Out.",    fmt_money(shares))

    st.markdown('<div class="eyebrow" style="margin-top:1rem;">Assumptions</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    growth_rate     = a1.slider("FCF Growth Rate",      0, 40, 10, 1, format="%d%%", key="dcf_g") / 100
    terminal_growth = a2.slider("Terminal Growth Rate", 0,  5,  2, 1, format="%d%%", key="dcf_t") / 100
    discount_rate   = a3.slider("Discount Rate (WACC)", 5, 20, 10, 1, format="%d%%", key="dcf_d") / 100
    proj_years      = a4.slider("Projection Years",     5, 15, 10,                   key="dcf_y")

    if discount_rate <= terminal_growth:
        st.error("Discount rate must exceed terminal growth rate.")
    else:
        intrinsic, cf_rows, tv_pv = run_dcf(base_fcf, base_rev, growth_rate, terminal_growth,
                                             discount_rate, proj_years, shares, net_debt)
        if intrinsic is None:
            st.warning("Not enough data — need FCF or Revenue plus share count.")
        else:
            mos = (intrinsic - price) / intrinsic if price else None
            mos_cls = "up" if mos and mos > 0 else "down"

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="dcf-card"><div class="dcf-label">Intrinsic Value / Share</div><div class="dcf-value">${intrinsic:.2f}</div></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="dcf-card"><div class="dcf-label">Current Price</div><div class="dcf-value">${price:.2f}</div></div>', unsafe_allow_html=True)
            with r3:
                if mos is not None:
                    lbl = "Upside" if mos > 0 else "Downside"
                    tag = "Appears undervalued" if mos > 0 else "Appears overvalued"
                    st.markdown(f'<div class="dcf-card"><div class="dcf-label">{lbl}</div><div class="dcf-value {mos_cls}">{mos*100:.1f}%</div><div class="dcf-delta {mos_cls}">{tag}</div></div>', unsafe_allow_html=True)

            df_cf = pd.DataFrame(cf_rows)
            fig_d = go.Figure(go.Bar(
                x=df_cf["Year"].tolist() + ["Terminal"],
                y=df_cf["PV ($M)"].tolist() + [tv_pv],
                marker_color=["#f5a623"] * len(df_cf) + ["#00e676"],
                marker_line_width=0,
            ))
            fig_d.update_layout(
                title=dict(text="Present Value of Cash Flows ($M)", font=dict(size=11, color="#5a6278")),
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Mono", color="#5a6278", size=10),
                margin=dict(l=0, r=0, t=35, b=0),
                xaxis=dict(showgrid=False, color="#5a6278"),
                yaxis=dict(showgrid=True, gridcolor="#1a1e28", color="#5a6278"),
            )
            st.plotly_chart(fig_d, use_container_width=True, config={"displayModeBar": False})

            st.markdown('<div class="eyebrow">Sensitivity — Intrinsic Value / Share</div>', unsafe_allow_html=True)
            g_range = [round(growth_rate + d, 2) for d in [-0.04, -0.02, 0, 0.02, 0.04]]
            d_range = [round(discount_rate + d, 3) for d in [-0.02, -0.01, 0, 0.01, 0.02]]
            sens = {}
            for dr in [d for d in d_range if d > terminal_growth and d > 0]:
                col_vals = []
                for gr in g_range:
                    iv, _, _ = run_dcf(base_fcf, base_rev, gr, terminal_growth, dr, proj_years, shares, net_debt)
                    col_vals.append(f"${iv:.0f}" if iv else "N/A")
                sens[f"WACC {dr*100:.1f}%"] = col_vals
            st.dataframe(pd.DataFrame(sens, index=[f"Growth {g*100:.0f}%" for g in g_range]),
                         use_container_width=True)
            st.caption("Rows = FCF growth rate · Columns = discount rate (WACC)")

# ── Peer Comparison ──
with tab7:
    st.markdown('<div class="eyebrow">Peer Comparison</div>', unsafe_allow_html=True)

    DEFAULT_PEERS = {
        "AAPL": "MSFT, GOOGL, META",   "MSFT": "AAPL, GOOGL, AMZN",
        "TSLA": "F, GM, RIVN",          "NVDA": "AMD, INTC, QCOM",
        "AMZN": "MSFT, GOOGL, WMT",    "GOOGL": "MSFT, META, AMZN",
        "META": "GOOGL, SNAP, PINS",   "JPM": "BAC, GS, MS",
        "JNJ":  "PFE, MRK, ABT",       "XOM": "CVX, BP, COP",
    }

    pc1, pc2 = st.columns([3, 1])
    with pc1:
        peer_input = st.text_input("peers", value=DEFAULT_PEERS.get(symbol, ""),
                                   key="peer_input", label_visibility="collapsed",
                                   placeholder="Enter peer tickers  e.g. MSFT, GOOGL, META")
    with pc2:
        run_peers = st.button("Compare ▸", type="secondary", key="btn_peers", use_container_width=True)

    if run_peers and peer_input:
        peers = [p.strip().upper() for p in peer_input.split(",") if p.strip()][:6]
        with st.spinner("Fetching peer data..."):
            peer_infos = {symbol: info}
            for p in peers:
                peer_infos[p] = fetch_peer(p)
        st.session_state["peer_infos"] = peer_infos

    peer_infos = st.session_state.get("peer_infos")

    if peer_infos:
        PEER_KEYS = [
            ("P/E",         "trailingPE",                   False),
            ("Fwd P/E",     "forwardPE",                    False),
            ("P/B",         "priceToBook",                  False),
            ("EV/EBITDA",   "enterpriseToEbitda",           False),
            ("Net Margin",  "profitMargins",                True),
            ("ROE",         "returnOnEquity",               True),
            ("Div Yield",   "dividendYield",                True),
            ("Rev Growth",  "revenueGrowth",                True),
            ("Beta",        "beta",                         False),
        ]
        rows = []
        for ticker, inf in peer_infos.items():
            row = {"Ticker": ticker,
                   "Company": (inf.get("longName") or ticker)[:20],
                   "Mkt Cap": fmt_money(inf.get("marketCap"))}
            for label, key, is_pct in PEER_KEYS:
                v = sg(inf, key)
                row[label] = (fmt_pct(v) if is_pct else fmt_num(v)) if v is not None else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

        # Bar charts
        st.markdown('<div class="eyebrow" style="margin-top:1.2rem;">Visual Comparison</div>', unsafe_allow_html=True)
        ccols = st.columns(2)
        for i, (label, key, is_pct) in enumerate(
            [m for m in PEER_KEYS if m[0] in ("P/E", "EV/EBITDA", "Net Margin", "ROE")]
        ):
            chart_data = []
            for ticker, inf in peer_infos.items():
                v = sg(inf, key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    chart_data.append({"Ticker": ticker, "Value": v * 100 if is_pct else v})
            if chart_data:
                cdf = pd.DataFrame(chart_data).sort_values("Value")
                colors = ["#f5a623" if t == symbol else "#252a38" for t in cdf["Ticker"]]
                fig_p = go.Figure(go.Bar(
                    x=cdf["Ticker"], y=cdf["Value"],
                    marker_color=colors,
                    marker_line_color=["#f5a623" if t == symbol else "#3a4055" for t in cdf["Ticker"]],
                    marker_line_width=1,
                ))
                fig_p.update_layout(
                    title=dict(text=f"{label}{' (%)' if is_pct else ''}", font=dict(size=11, color="#5a6278")),
                    height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Mono", color="#5a6278", size=10),
                    showlegend=False, margin=dict(l=0, r=0, t=35, b=0),
                    xaxis=dict(showgrid=False, color="#5a6278"),
                    yaxis=dict(showgrid=True, gridcolor="#1a1e28", color="#5a6278"),
                )
                ccols[i % 2].plotly_chart(fig_p, use_container_width=True,
                                          config={"displayModeBar": False})

        # Percentile ranking
        st.markdown(f'<div class="eyebrow" style="margin-top:1rem;">{symbol} Percentile vs Peers</div>', unsafe_allow_html=True)
        pct_rows = []
        for label, key, is_pct in PEER_KEYS:
            vals = {t: sg(inf, key) for t, inf in peer_infos.items() if sg(inf, key) is not None}
            if symbol in vals and len(vals) > 1:
                sv   = sorted(vals.values())
                rank = sv.index(vals[symbol]) + 1
                n    = len(sv)
                pr   = (rank / n * 100) if is_pct else ((n - rank + 1) / n * 100)
                interp = "✅ Better than most" if pr >= 60 else ("⚠️ Middle of pack" if pr >= 40 else "🔴 Lags peers")
                pct_rows.append({
                    "Metric":    label,
                    symbol:      fmt_pct(vals[symbol]) if is_pct else fmt_num(vals[symbol]),
                    "Percentile":f"{pr:.0f}th",
                    "vs Peers":  interp,
                })
        if pct_rows:
            st.dataframe(pd.DataFrame(pct_rows), hide_index=True, use_container_width=True)
    else:
        st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.8rem;color:#5a6278;padding:2rem 0;">Enter peer tickers above and click Compare ▸</div>', unsafe_allow_html=True)
