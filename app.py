"""
Automated Market Intelligence Dashboard
----------------------------------------
Ticker in -> price snapshot, an interactive candlestick chart with support
/ resistance and options open-interest "walls", options positioning, and
analyst price targets.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API keys:
    1. Finnhub (quote, company name, next earnings) — free, no credit card,
       60 calls/minute.  https://finnhub.io/register
       -> FINNHUB_API_KEY   (env var, or Streamlit Cloud Secrets)   [required]

    2. Twelve Data (historical daily prices for the candlestick) — free,
       no credit card, 800 calls/day.  https://twelvedata.com/pricing
       -> TWELVEDATA_API_KEY   (env var, or Streamlit Cloud Secrets) [required]

    3. Financial Modeling Prep (analyst price targets) — free tier.
       https://site.financialmodelingprep.com/pricing-plans
       -> FMP_API_KEY   (env var, or Streamlit Cloud Secrets)        [optional;
          without it, analyst targets fall back to Yahoo Finance]

    Options data (open-interest walls) comes from yfinance (Yahoo) — no key.
"""

import os
import statistics
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ============================================================================
# CONFIG + THEME
# ============================================================================

st.set_page_config(page_title="Market Intelligence Dashboard", page_icon="📈", layout="wide")

# Dark palette: black background, light text, terracotta accent, bold Inter
# headings. Card/panel surfaces sit slightly above pure black.
ACCENT = "#FFFFFF"       # accent: white (metric values, SMA line, target bars)
INK = "#F5F3EF"          # primary (light) text on black
MUTED = "#9A9186"        # secondary text, readable on black
CARD_BG = "#141414"      # metric card / panel surface
CARD_BORDER = "#2A2A2A"
CHART_BG = "#0A0A0A"     # chart plot area
GRID = "#242424"
SUPPORT_GREEN = "#37C978"
RESIST_RED = "#F0603C"
CALL_PURPLE = "#B26BE0"
PUT_BLUE = "#3FA3F0"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
    .stApp {{ background: #000000; }}
    html, body, [class*="css"], .stMarkdown, .stMetric {{ font-family: 'Inter', sans-serif; }}
    h1, h2, h3 {{ font-weight: 800 !important; color: {INK}; letter-spacing: -0.02em; }}
    [data-testid="stMetric"] {{
        background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 14px;
        padding: 14px 18px;
    }}
    [data-testid="stMetricValue"] {{ color: {ACCENT}; font-weight: 800; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; text-transform: uppercase;
        letter-spacing: .05em; font-size: .72rem; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: .14em; font-size: .72rem;
        font-weight: 700; color: {MUTED}; margin-bottom: .2rem; }}
    .hero-title {{ font-size: 2.6rem; font-weight: 800; line-height: 1.05;
        letter-spacing: -0.02em; color: {INK}; }}
    .hero-sub {{ color: {MUTED}; font-weight: 600; margin-top: .1rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Peer groups for the sector-average P/E. Matched by substring against
# Finnhub's `finnhubIndustry` string when the dynamic /stock/peers call
# returns nothing.
INDUSTRY_KEYWORD_PEERS = [
    (["semiconductor"], ["AMD", "INTC", "QCOM", "TXN"]),
    (["bank"], ["JPM", "BAC", "C", "WFC"]),
    (["software"], ["MSFT", "ORCL", "CRM", "ADBE"]),
    (["internet", "e-commerce", "ecommerce"], ["AMZN", "EBAY", "ETSY", "SHOP"]),
    (["media", "entertainment"], ["GOOGL", "META", "NFLX", "DIS"]),
    (["auto", "vehicle"], ["TSLA", "GM", "F", "TM"]),
    (["oil", "gas", "energy"], ["XOM", "CVX", "COP", "SLB"]),
    (["pharma", "biotech", "drug"], ["PFE", "MRK", "JNJ", "ABBV"]),
    (["insurance"], ["UNH", "CI", "ELV", "HUM"]),
    (["airline"], ["DAL", "UAL", "AAL", "LUV"]),
    (["aerospace", "defense"], ["BA", "LMT", "RTX", "NOC"]),
    (["telecom"], ["T", "VZ", "TMUS", "CMCSA"]),
    (["hardware", "electronics", "computer"], ["AAPL", "MSFT", "GOOGL", "HPQ"]),
    (["retail"], ["WMT", "TGT", "COST", "HD"]),
    (["real estate", "reit"], ["PLD", "AMT", "EQIX", "SPG"]),
    (["utilit"], ["NEE", "DUK", "SO", "D"]),
    (["metal", "mining", "material"], ["LIN", "SHW", "FCX", "NEM"]),
]
DEFAULT_PEERS = ["MSFT", "AAPL", "GOOGL", "AMZN"]
PEER_LIMIT = 20          # top-N peers by market cap for the median
PE_OUTLIER_MAX = 200.0   # discard P/E above this as an extreme outlier


# ============================================================================
# API KEYS
# ============================================================================


def _get_key(name: str):
    """Read an API key from an env var, falling back to Streamlit secrets."""
    val = os.environ.get(name)
    if not val:
        try:
            val = st.secrets[name]
        except Exception:
            val = None
    return val


def _with_retry(func, retries: int = 3, base_delay: float = 2.0):
    """Retry an API call with exponential backoff (free tiers throttle)."""
    last_exc = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc


# ============================================================================
# DATA FETCHING
# ============================================================================


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_core(ticker: str, api_key: str):
    """Quote, profile, fundamentals (P/E, EPS), and next earnings from Finnhub."""
    client = finnhub.Client(api_key=api_key)

    def _load():
        p = client.company_profile2(symbol=ticker)
        q = client.quote(ticker)
        if not p or not q or not q.get("c"):
            raise RuntimeError("Empty response from Finnhub (invalid ticker or rate limit)")
        return p, q

    # Propagate so the real error surfaces and a failed call isn't cached.
    profile, quote = _with_retry(_load)

    try:
        metrics = client.company_basic_financials(ticker, "all") or {}
    except Exception:
        metrics = {}

    end = datetime.now()
    try:
        earnings = (
            client.earnings_calendar(
                _from=end.strftime("%Y-%m-%d"),
                to=(end + timedelta(days=120)).strftime("%Y-%m-%d"),
                symbol=ticker,
            )
            or {}
        )
    except Exception:
        earnings = {}
    return profile, quote, metrics, earnings


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str, api_key: str) -> pd.DataFrame:
    """Daily OHLCV from Twelve Data (Finnhub's candles are paid-only)."""

    def _load():
        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": ticker, "interval": "1day", "outputsize": 180, "apikey": api_key},
            timeout=15,
        )
        data = resp.json()
        if data.get("status") != "ok" or not data.get("values"):
            raise RuntimeError(data.get("message", "Empty response from Twelve Data"))
        return data["values"]

    values = _with_retry(_load)
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    return df.set_index("datetime").sort_index()[["Open", "High", "Low", "Close", "Volume"]]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_expirations(ticker: str) -> list:
    """Available option expiry dates from yfinance (US equities), newest first."""
    try:
        exps = _with_retry(lambda: yf.Ticker(ticker).options, retries=2, base_delay=1.5)
        return list(exps) if exps else []
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_walls(ticker: str, expiry: str) -> dict:
    """
    Options positioning for one expiry: the strike with the most call open
    interest ("call wall") and the strike with the most put open interest
    ("put wall") — i.e. where most calls and puts are concentrated — plus a
    put/call open-interest ratio.
    """
    try:
        chain = _with_retry(lambda: yf.Ticker(ticker).option_chain(expiry), retries=2, base_delay=1.5)
    except Exception:
        return {}
    calls, puts = chain.calls, chain.puts
    out = {}
    if calls is not None and not calls.empty and calls["openInterest"].notna().any():
        row = calls.loc[calls["openInterest"].idxmax()]
        out["call_wall"] = float(row["strike"])
        out["call_wall_oi"] = int(row["openInterest"])
        out["call_oi_total"] = int(calls["openInterest"].fillna(0).sum())
    if puts is not None and not puts.empty and puts["openInterest"].notna().any():
        row = puts.loc[puts["openInterest"].idxmax()]
        out["put_wall"] = float(row["strike"])
        out["put_wall_oi"] = int(row["openInterest"])
        out["put_oi_total"] = int(puts["openInterest"].fillna(0).sum())
    if out.get("call_oi_total") and out.get("put_oi_total"):
        out["put_call_ratio"] = out["put_oi_total"] / out["call_oi_total"]
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analyst_targets(ticker: str, fmp_key: str) -> dict:
    """
    Analyst price-target consensus (low / mean / high). Financial Modeling
    Prep first (the user's chosen source), then a yfinance fallback.
    """
    if fmp_key:
        try:
            resp = _with_retry(
                lambda: requests.get(
                    "https://financialmodelingprep.com/stable/price-target-consensus",
                    params={"symbol": ticker, "apikey": fmp_key},
                    timeout=15,
                ),
                retries=2,
                base_delay=1.5,
            )
            data = resp.json()
            row = data[0] if isinstance(data, list) and data else {}
            if row.get("targetConsensus"):
                return {
                    "low": row.get("targetLow"),
                    "mean": row.get("targetConsensus"),
                    "high": row.get("targetHigh"),
                    "count": None,
                    "source": "Financial Modeling Prep",
                }
        except Exception:
            pass
    # yfinance (Yahoo) fallback.
    try:
        info = _with_retry(lambda: yf.Ticker(ticker).info, retries=2, base_delay=1.5) or {}
        if info.get("targetMeanPrice"):
            return {
                "low": info.get("targetLowPrice"),
                "mean": info.get("targetMeanPrice"),
                "high": info.get("targetHighPrice"),
                "count": info.get("numberOfAnalystOpinions"),
                "source": "Yahoo Finance",
            }
    except Exception:
        pass
    return None


def get_next_earnings_date(earnings: dict) -> str:
    try:
        entries = earnings.get("earningsCalendar") or []
        dates = sorted(e.get("date") for e in entries if e.get("date"))
        return dates[0] if dates else "N/A"
    except Exception:
        return "N/A"


# ============================================================================
# FUNDAMENTALS (P/E, EPS, sector-average P/E, fair value)
# ============================================================================


def metric_value(metrics: dict, *names):
    """First present numeric value among candidate Finnhub metric keys."""
    m = (metrics or {}).get("metric") or {}
    for n in names:
        v = m.get(n)
        if isinstance(v, (int, float)):
            return v
    return None


def get_sector_peers_static(industry: str, exclude_ticker: str) -> list:
    """Fallback peer group when Finnhub's /stock/peers returns nothing."""
    industry_lower = (industry or "").lower()
    for keywords, peers in INDUSTRY_KEYWORD_PEERS:
        if any(kw in industry_lower for kw in keywords):
            return [p for p in peers if p.upper() != exclude_ticker.upper()]
    return [p for p in DEFAULT_PEERS if p.upper() != exclude_ticker.upper()]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peer_group(ticker: str, industry: str, api_key: str) -> list:
    """Sub-industry peer group from Finnhub /stock/peers, with a static fallback."""
    client = finnhub.Client(api_key=api_key)
    try:
        peers = _with_retry(lambda: client.company_peers(ticker), retries=2, base_delay=1.5) or []
    except Exception:
        peers = []
    peers = [p for p in peers if p and p.upper() != ticker.upper()]
    if not peers:
        peers = get_sector_peers_static(industry, ticker)
    return peers[:PEER_LIMIT]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peer_pes(peers: list, api_key: str) -> list:
    """Per-peer trailing P/E and market cap for the sector-average calc."""
    client = finnhub.Client(api_key=api_key)
    rows = []
    for peer in peers:
        try:
            data = _with_retry(
                lambda p=peer: client.company_basic_financials(p, "all"), retries=2, base_delay=1.5
            )
            metric = data.get("metric") or {}
            rows.append({"ticker": peer, "pe": metric.get("peTTM"),
                         "market_cap": metric.get("marketCapitalization")})
        except Exception:
            continue
    return rows


def compute_sector_pe(peer_rows: list) -> dict:
    """
    Sector-average P/E via the MEDIAN of cleansed peers (a plain mean gets
    skewed by mega-cap outliers), after dropping negative/zero P/E and P/E
    above the outlier cap, keeping the top-N by market cap.
    """
    valid = [r for r in peer_rows
             if isinstance(r.get("pe"), (int, float)) and 0 < r["pe"] <= PE_OUTLIER_MAX]
    if any(r.get("market_cap") for r in valid):
        valid = sorted(valid, key=lambda r: r.get("market_cap") or 0, reverse=True)[:PEER_LIMIT]
    pes = [r["pe"] for r in valid]
    return {
        "median_pe": statistics.median(pes) if pes else None,
        "mean_pe": statistics.fmean(pes) if pes else None,
        "valid_peers": valid,
        "count": len(pes),
    }


def fair_value_from_sector_pe(sector_pe, eps):
    """Relative fair value = sector-average P/E × company EPS."""
    if not sector_pe or eps is None:
        return None
    return sector_pe * eps


# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================


def sma_series(hist: pd.DataFrame, window: int = 50) -> pd.Series:
    return hist["Close"].rolling(window=window).mean()


def calculate_sma(hist: pd.DataFrame, window: int = 50):
    if hist.empty or len(hist) < window:
        return None
    return hist["Close"].rolling(window=window).mean().iloc[-1]


def determine_trend(current_price, sma) -> str:
    if sma is None or current_price is None:
        return "Unknown"
    return "Uptrend" if current_price > sma else "Downtrend"


def calculate_pivot_points(hist: pd.DataFrame, lookback_days: int = 30):
    """Standard pivot points from the High/Low/Close of the trailing window."""
    if hist.empty:
        return None
    window = hist.tail(lookback_days)
    if window.empty:
        return None
    high = window["High"].max()
    low = window["Low"].min()
    close = window["Close"].iloc[-1]
    pivot = (high + low + close) / 3
    return {
        "pivot": pivot,
        "s1": (2 * pivot) - high,
        "r1": (2 * pivot) - low,
        "high": high,
        "low": low,
    }


def build_candlestick(hist: pd.DataFrame, sma, pivots, walls, current_price):
    """Interactive candlestick with SMA, support/resistance, and option walls."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
                name="Price",
                increasing_line_color=SUPPORT_GREEN,
                decreasing_line_color=RESIST_RED,
            )
        ]
    )
    if sma is not None:
        fig.add_trace(
            go.Scatter(
                x=hist.index, y=sma, name="50-day SMA",
                line=dict(color=ACCENT, width=1.6),
            )
        )

    def hline(y, color, text):
        if y is not None:
            fig.add_hline(
                y=y, line_dash="dash", line_color=color, line_width=1.3, opacity=0.85,
                annotation_text=text, annotation_position="top left",
                annotation_font_color=color, annotation_font_size=11,
            )

    if pivots:
        hline(pivots["r1"], RESIST_RED, "R1 resistance")
        hline(pivots["s1"], SUPPORT_GREEN, "S1 support")
    if walls:
        if walls.get("call_wall"):
            hline(walls["call_wall"], CALL_PURPLE, f"Call wall {walls['call_wall']:g}")
        if walls.get("put_wall"):
            hline(walls["put_wall"], PUT_BLUE, f"Put wall {walls['put_wall']:g}")

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=CHART_BG,
        font=dict(family="Inter, sans-serif", color=INK),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor=GRID),
        yaxis=dict(gridcolor=GRID),
    )
    return fig


# ============================================================================
# UI — SIDEBAR + KEY GATE
# ============================================================================

st.sidebar.markdown('<div class="eyebrow">Market Intelligence</div>', unsafe_allow_html=True)
ticker_input = st.sidebar.text_input("Stock ticker", value="NVDA").upper().strip()
st.sidebar.button("Analyze", type="primary", use_container_width=True)
st.sidebar.caption("Data: Finnhub · Twelve Data · Financial Modeling Prep · Yahoo Finance (options).")

if not ticker_input:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

finnhub_key = _get_key("FINNHUB_API_KEY")
twelvedata_key = _get_key("TWELVEDATA_API_KEY")
fmp_key = _get_key("FMP_API_KEY")  # optional (analyst targets fall back to Yahoo)

missing = []
if not finnhub_key:
    missing.append("FINNHUB_API_KEY (get one at https://finnhub.io/register)")
if not twelvedata_key:
    missing.append("TWELVEDATA_API_KEY (get one at https://twelvedata.com/pricing)")
if missing:
    st.error(
        "Missing required API key(s):\n\n"
        + "\n".join(f"- {k}" for k in missing)
        + "\n\nSet these as environment variables or Streamlit secrets."
    )
    st.stop()

with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        profile, quote, metrics, earnings = fetch_core(ticker_input, finnhub_key)
        hist = fetch_price_history(ticker_input, twelvedata_key)
    except Exception as e:
        st.error(f"Could not fetch data for '{ticker_input}': {e}")
        st.stop()

if not quote or not quote.get("c") or hist.empty:
    st.error(
        f"No data returned for '{ticker_input}'. Either the symbol is wrong, or a "
        "data source is rate-limiting — if the ticker is correct, wait a moment "
        "and click Analyze again."
    )
    st.stop()

current_price = quote.get("c")
company_name = profile.get("name", ticker_input)
exchange = profile.get("exchange", "")
previous_close = quote.get("pc", current_price)
price_change = current_price - previous_close if previous_close else 0
price_change_pct = (price_change / previous_close * 100) if previous_close else 0
next_earnings = get_next_earnings_date(earnings)

sma_50 = calculate_sma(hist, 50)
sma_line = sma_series(hist, 50)
trend = determine_trend(current_price, sma_50)
pivots = calculate_pivot_points(hist, 30)

# ============================================================================
# UI — HERO
# ============================================================================

st.markdown('<div class="eyebrow">Equity Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-title">{company_name}</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="hero-sub">{ticker_input}{" · " + exchange if exchange else ""}</div>',
    unsafe_allow_html=True,
)
st.write("")

h1, h2, h3, h4 = st.columns(4)
h1.metric("Price", f"${current_price:,.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
h2.metric("Trend", trend, f"{(current_price - sma_50):+.2f} vs 50-SMA" if sma_50 else None,
          delta_color="off")
h3.metric("50-Day SMA", f"${sma_50:,.2f}" if sma_50 else "N/A")
h4.metric("Next Earnings", next_earnings)

st.divider()

# ============================================================================
# UI — OPTIONS EXPIRY PICKER (drives the chart's walls + the options section)
# ============================================================================

expirations = fetch_option_expirations(ticker_input)
selected_expiry = None
walls = {}
if expirations:
    selected_expiry = st.selectbox(
        "Options expiry (for open-interest walls)", expirations, index=0, key="expiry"
    )
    walls = fetch_option_walls(ticker_input, selected_expiry)

# ============================================================================
# UI — CANDLESTICK CHART
# ============================================================================

st.header("Price Chart — Candles, Support/Resistance & Option Walls")
fig = build_candlestick(hist, sma_line, pivots, walls, current_price)
st.plotly_chart(fig, use_container_width=True)

legend_bits = ["🟠 50-SMA"]
if pivots:
    legend_bits += [f"🟢 S1 support ${pivots['s1']:,.2f}", f"🔴 R1 resistance ${pivots['r1']:,.2f}"]
if walls.get("call_wall"):
    legend_bits.append(f"🟣 Call wall ${walls['call_wall']:,.2f}")
if walls.get("put_wall"):
    legend_bits.append(f"🔵 Put wall ${walls['put_wall']:,.2f}")
st.caption(" · ".join(legend_bits))

st.divider()

# ============================================================================
# UI — OPTIONS POSITIONING
# ============================================================================

st.header("Options Positioning — Where Most Puts & Calls Sit")
if not expirations:
    st.info(
        "No options data available for this ticker (yfinance/Yahoo returned no "
        "expiries — the symbol may not have listed options, or Yahoo is rate-limiting)."
    )
elif not walls:
    st.info(f"Open-interest data unavailable for the {selected_expiry} expiry.")
else:
    o1, o2, o3 = st.columns(3)
    if walls.get("call_wall"):
        gap = (walls["call_wall"] - current_price) / current_price * 100
        o1.metric("Call Wall (max OI strike)", f"${walls['call_wall']:,.2f}", f"{gap:+.1f}% vs price")
        o1.caption(f"{walls['call_wall_oi']:,} contracts of open interest — acts as resistance.")
    if walls.get("put_wall"):
        gap = (walls["put_wall"] - current_price) / current_price * 100
        o2.metric("Put Wall (max OI strike)", f"${walls['put_wall']:,.2f}", f"{gap:+.1f}% vs price")
        o2.caption(f"{walls['put_wall_oi']:,} contracts of open interest — acts as support.")
    if walls.get("put_call_ratio") is not None:
        pcr = walls["put_call_ratio"]
        lean = "put-heavy (bearish)" if pcr > 1 else "call-heavy (bullish)"
        o3.metric("Put/Call OI Ratio", f"{pcr:.2f}", lean, delta_color="off")
        o3.caption(f"Expiry {selected_expiry}.")

st.divider()

# ============================================================================
# UI — FUNDAMENTALS (P/E, EPS, sector-average P/E, fair value)
# ============================================================================

st.header("Fundamentals — Relative Fair Value")

trailing_pe = metric_value(metrics, "peTTM")
trailing_eps = metric_value(metrics, "epsTTM")
peers = fetch_peer_group(ticker_input, profile.get("finnhubIndustry"), finnhub_key)
sector = compute_sector_pe(fetch_peer_pes(peers, finnhub_key))
sector_pe = sector["median_pe"]
fair_value = fair_value_from_sector_pe(sector_pe, trailing_eps)

fcol1, fcol2, fcol3 = st.columns(3)
fcol1.metric("P/E Ratio (trailing)", f"{trailing_pe:.1f}x" if trailing_pe else "N/A",
             help="Price ÷ EPS — what you pay per $1 of trailing profit.")
fcol2.metric("EPS (trailing, TTM)", f"${trailing_eps:,.2f}" if trailing_eps is not None else "N/A",
             help="Earnings per share over the trailing twelve months.")
fcol3.metric("Sector Avg P/E", f"{sector_pe:.1f}x" if sector_pe else "N/A",
             f"{sector['count']} peers" if sector["count"] else None, delta_color="off",
             help="Median (not mean) of cleansed sub-industry peers — resistant to outliers.")

if fair_value and fair_value > 0:
    premium_discount = (current_price - fair_value) / fair_value * 100
    label = "premium to" if premium_discount > 0 else "discount to"
    st.metric("Implied Fair Value  (Sector Avg P/E × EPS)", f"${fair_value:,.2f}",
              f"{premium_discount:+.1f}% ({label} fair value)", delta_color="inverse")
    st.caption(
        f"Fair value = sector avg P/E **{sector_pe:.1f}x** × EPS **${trailing_eps:,.2f}** "
        f"= **${fair_value:,.2f}**, vs current price ${current_price:,.2f}."
    )
else:
    st.metric("Implied Fair Value  (Sector Avg P/E × EPS)", "N/A")
    st.caption(
        "Needs both a sector average P/E and a positive company EPS — one wasn't "
        "available for this ticker (e.g. the company is unprofitable, or peer data "
        "was rate-limited)."
    )

with st.expander(f"Peer group used for the sector average ({', '.join(peers) if peers else 'none'})"):
    valid = sector["valid_peers"]
    if valid:
        peer_df = pd.DataFrame(
            [{"Ticker": r["ticker"], "Trailing P/E": round(r["pe"], 1)} for r in valid]
        )
        st.dataframe(peer_df, hide_index=True, use_container_width=True)
        if sector["mean_pe"] and sector_pe:
            st.caption(
                f"Median P/E **{sector_pe:.1f}x** vs mean **{sector['mean_pe']:.1f}x** — the "
                "median is used to resist mega-cap outliers. Negative and >200 P/E peers filtered out."
            )
    else:
        st.write("No valid peer P/E data returned.")

st.divider()

# ============================================================================
# UI — ANALYST PRICE TARGETS
# ============================================================================

st.header("Analyst Price Targets (next 12 months)")
targets = fetch_analyst_targets(ticker_input, fmp_key)
if targets and isinstance(targets.get("mean"), (int, float)):
    tcols = st.columns(3)
    for col, key, label in zip(tcols, ["low", "mean", "high"], ["Min", "Avg", "Max"]):
        val = targets.get(key)
        if isinstance(val, (int, float)) and val > 0:
            upside = (val - current_price) / current_price * 100
            col.metric(f"{label} Target", f"${val:,.2f}", f"{upside:+.1f}% vs current")
        else:
            col.metric(f"{label} Target", "N/A")
    n = targets.get("count")
    st.caption(f"Source: {targets.get('source')}" + (f" · {n} analysts" if n else ""))

    chart_df = pd.DataFrame(
        {"Price ($)": {
            "Min": targets.get("low"), "Current": current_price,
            "Avg": targets.get("mean"), "Max": targets.get("high"),
        }}
    )
    chart_df = chart_df[chart_df["Price ($)"].apply(lambda v: isinstance(v, (int, float)) and v > 0)]
    if not chart_df.empty:
        st.bar_chart(chart_df, color=ACCENT)
else:
    st.info(
        "No analyst price targets available. Set FMP_API_KEY (free at "
        "financialmodelingprep.com) for consensus targets; the Yahoo Finance "
        "fallback returned none (it may be rate-limited or the ticker isn't covered)."
    )

st.divider()
st.caption("Data: Finnhub, Twelve Data, Financial Modeling Prep, Yahoo Finance. Not investment advice.")
