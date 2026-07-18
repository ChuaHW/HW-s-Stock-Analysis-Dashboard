"""
Automated Market Intelligence Dashboard
----------------------------------------
Takes a stock ticker and produces a technical, fundamental, and
qualitative (LLM-powered) analysis in a single Streamlit view.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API key:
    This app calls the Anthropic (Claude) API for the "Market Narrative"
    section. See `get_llm_client()` below for exactly where to put your key.
    - Local dev:      set the ANTHROPIC_API_KEY environment variable
    - Streamlit Cloud: Advanced settings -> Secrets ->
                            ANTHROPIC_API_KEY = "sk-ant-..."
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
import anthropic

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Automated Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
)

LLM_MODEL = "claude-opus-4-8"

# Industry -> 3-4 comparable peer tickers. Keys line up with yfinance's
# `industry` field where possible; SECTOR_PEER_FALLBACK below covers
# anything not explicitly listed here.
INDUSTRY_PEER_MAP = {
    "Semiconductors": ["AMD", "INTC", "QCOM", "TXN"],
    "Semiconductor Equipment & Materials": ["ASML", "AMAT", "LRCX", "KLAC"],
    "Software - Infrastructure": ["MSFT", "ORCL", "CRM", "ADBE"],
    "Software - Application": ["CRM", "ADBE", "INTU", "NOW"],
    "Banks - Diversified": ["JPM", "BAC", "C", "WFC"],
    "Banks - Regional": ["USB", "PNC", "TFC", "COF"],
    "Internet Retail": ["AMZN", "EBAY", "ETSY", "SHOP"],
    "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS"],
    "Auto Manufacturers": ["TSLA", "GM", "F", "TM"],
    "Oil & Gas Integrated": ["XOM", "CVX", "SHEL", "BP"],
    "Drug Manufacturers - General": ["PFE", "MRK", "JNJ", "ABBV"],
    "Credit Services": ["V", "MA", "AXP", "PYPL"],
    "Consumer Electronics": ["AAPL", "SONY", "HPQ", "DELL"],
    "Airlines": ["DAL", "UAL", "AAL", "LUV"],
    "Aerospace & Defense": ["BA", "LMT", "RTX", "NOC"],
}

SECTOR_PEER_FALLBACK = {
    "Technology": ["MSFT", "AAPL", "GOOGL", "META"],
    "Financial Services": ["JPM", "BAC", "V", "MA"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Industrials": ["HON", "UNP", "CAT", "GE"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT"],
    "Utilities": ["NEE", "DUK", "SO", "D"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG"],
    "Basic Materials": ["LIN", "SHW", "FCX", "NEM"],
}

DEFAULT_PEERS = ["MSFT", "AAPL", "GOOGL", "AMZN"]


# ============================================================================
# LLM CLIENT
# ============================================================================


def get_llm_client():
    """
    Returns an anthropic.Anthropic client, or None if no key is configured.

    >>> INSERT YOUR API KEY <<<
    Set it as the environment variable ANTHROPIC_API_KEY, or (on Streamlit
    Community Cloud) add ANTHROPIC_API_KEY under Advanced Settings -> Secrets.
    Never hardcode the key directly in this file.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["ANTHROPIC_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return anthropic.Anthropic(api_key=api_key)


def call_claude(client, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    if client is None:
        return "LLM analysis unavailable — no ANTHROPIC_API_KEY configured."
    try:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
    except Exception as e:
        return f"LLM request failed: {e}"


# ============================================================================
# DATA FETCHING (yfinance)
# ============================================================================


@st.cache_data(ttl=300, show_spinner=False)
def fetch_ticker_data(ticker: str):
    """Pulls everything needed from yfinance in one cached call."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    hist = tk.history(period="3mo")
    try:
        news = tk.news or []
    except Exception:
        news = []
    try:
        calendar = tk.calendar
    except Exception:
        calendar = None
    return info, hist, news, calendar


@st.cache_data(ttl=300, show_spinner=False)
def fetch_peer_pe_ratios(peers: list) -> dict:
    pe_data = {}
    for peer in peers:
        try:
            peer_info = yf.Ticker(peer).info
            pe = peer_info.get("trailingPE")
            if pe and pe > 0:
                pe_data[peer] = pe
        except Exception:
            continue
    return pe_data


def get_next_earnings_date(calendar) -> str:
    if calendar is None:
        return "N/A"
    try:
        if isinstance(calendar, dict):
            dates = calendar.get("Earnings Date")
        else:  # older yfinance returns a DataFrame
            dates = calendar.loc["Earnings Date"].values if "Earnings Date" in calendar.index else None
        if not dates:
            return "N/A"
        date_val = dates[0]
        if hasattr(date_val, "strftime"):
            return date_val.strftime("%Y-%m-%d")
        return str(date_val)
    except Exception:
        return "N/A"


# ============================================================================
# NEWS SCHEMA HELPERS
# yfinance has shipped a couple of different shapes for `Ticker.news` over
# time. These helpers normalize both so the rest of the app doesn't care.
# ============================================================================


def get_news_title(item: dict) -> str:
    return item.get("title") or (item.get("content") or {}).get("title", "")


def get_news_publisher(item: dict) -> str:
    publisher = item.get("publisher")
    if publisher:
        return publisher
    provider = (item.get("content") or {}).get("provider") or {}
    return provider.get("displayName", "")


def get_news_datetime(item: dict):
    ts = item.get("providerPublishTime")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    pub_date = (item.get("content") or {}).get("pubDate")
    if pub_date:
        try:
            return pd.to_datetime(pub_date).tz_localize(None).to_pydatetime()
        except Exception:
            return None
    return None


def format_headlines(news_items: list) -> str:
    lines = []
    for item in news_items[:5]:
        title = get_news_title(item)
        publisher = get_news_publisher(item)
        if title:
            lines.append(f"- {title} ({publisher})" if publisher else f"- {title}")
    return "\n".join(lines) if lines else "No headlines available."


# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================


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
    s1 = (2 * pivot) - high
    r1 = (2 * pivot) - low
    return {"pivot": pivot, "s1": s1, "r1": r1, "high": high, "low": low, "close": close}


# ============================================================================
# FUNDAMENTAL ANALYSIS
# ============================================================================


def get_sector_peers(info: dict, exclude_ticker: str) -> list:
    industry = info.get("industry")
    sector = info.get("sector")
    peers = INDUSTRY_PEER_MAP.get(industry) or SECTOR_PEER_FALLBACK.get(sector) or DEFAULT_PEERS
    return [p for p in peers if p.upper() != exclude_ticker.upper()][:4]


def calculate_fair_value(sector_avg_pe, target_eps):
    if not sector_avg_pe or target_eps is None:
        return None
    return sector_avg_pe * target_eps


# ============================================================================
# QUALITATIVE ANALYSIS
# ============================================================================


def find_largest_move_date(hist: pd.DataFrame, lookback_days: int = 30):
    recent = hist.tail(lookback_days).copy()
    if len(recent) < 2:
        return None, None
    recent["pct_change"] = recent["Close"].pct_change() * 100
    recent = recent.dropna(subset=["pct_change"])
    if recent.empty:
        return None, None
    idx = recent["pct_change"].abs().idxmax()
    return idx.date(), recent.loc[idx, "pct_change"]


def get_news_for_date(news: list, target_date) -> list:
    matched = []
    for item in news:
        dt = get_news_datetime(item)
        if dt and dt.date() == target_date:
            matched.append(item)
    return matched


def get_recent_news(news: list, hours: int = 48) -> list:
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = []
    for item in news:
        dt = get_news_datetime(item)
        if dt and dt >= cutoff:
            recent.append(item)
    return recent


def get_catalyst_analysis(client, ticker, move_date, pct_change, news_items) -> str:
    headlines = format_headlines(news_items)
    prompt = (
        f"Stock: {ticker}\n"
        f"Date of move: {move_date}\n"
        f"Price change: {pct_change:.2f}%\n\n"
        f"Headlines from that date:\n{headlines}\n\n"
        "Explain in 2 sentences exactly why this stock moved significantly "
        "on this date based on these headlines."
    )
    return call_claude(client, "You are a concise financial analyst.", prompt, max_tokens=200)


def get_current_narrative(client, ticker, news_items) -> str:
    headlines = format_headlines(news_items)
    prompt = (
        f"Stock: {ticker}\n\n"
        f"Headlines from the last 48 hours:\n{headlines}\n\n"
        "Synthesize the current market sentiment and narrative for this "
        "stock into a tight, 3-sentence summary."
    )
    return call_claude(client, "You are a concise financial analyst.", prompt, max_tokens=250)


# ============================================================================
# UI — SIDEBAR
# ============================================================================

st.sidebar.title("Market Intelligence")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper().strip()
st.sidebar.button("Analyze", type="primary", use_container_width=True)
st.sidebar.caption("Data: Yahoo Finance (yfinance). Narrative: Claude API.")

st.title("📈 Automated Market Intelligence Dashboard")

if not ticker_input:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        info, hist, news, calendar = fetch_ticker_data(ticker_input)
    except Exception as e:
        st.error(f"Could not fetch data for '{ticker_input}': {e}")
        st.stop()

if hist.empty or not info:
    st.error(f"No data found for ticker '{ticker_input}'. Please check the symbol.")
    st.stop()

current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
company_name = info.get("longName", ticker_input)
previous_close = info.get("previousClose", current_price)
price_change = current_price - previous_close if previous_close else 0
price_change_pct = (price_change / previous_close * 100) if previous_close else 0
next_earnings = get_next_earnings_date(calendar)

# ============================================================================
# UI — HEADER
# ============================================================================

st.subheader(f"{company_name} ({ticker_input})")

col1, col2, col3 = st.columns(3)
col1.metric(
    "Current Price",
    f"${current_price:,.2f}",
    f"{price_change:+.2f} ({price_change_pct:+.2f}%)",
)
col2.metric("Company", company_name)
col3.metric("Next Earnings Date", next_earnings)

st.divider()

# ============================================================================
# UI — TECHNICAL ANALYSIS
# ============================================================================

st.header("Technical Analysis")

sma_50 = calculate_sma(hist, window=50)
trend = determine_trend(current_price, sma_50)
pivots = calculate_pivot_points(hist, lookback_days=30)

tcol1, tcol2, tcol3 = st.columns(3)

with tcol1:
    delta_val = (current_price - sma_50) if sma_50 else 0
    st.metric("Trend (vs 50-Day SMA)", trend, f"{delta_val:+.2f} vs SMA")

with tcol2:
    st.metric("50-Day SMA", f"${sma_50:,.2f}" if sma_50 else "N/A")

with tcol3:
    if pivots:
        if trend == "Uptrend":
            gap = pivots["r1"] - current_price
            st.metric("Next Resistance (R1)", f"${pivots['r1']:,.2f}", f"{gap:+.2f} to clear")
        else:
            gap = pivots["s1"] - current_price
            st.metric("Next Support (S1)", f"${pivots['s1']:,.2f}", f"{gap:+.2f} to test")
    else:
        st.metric("Pivot Level", "N/A")

with st.expander("Pivot point detail (trailing 30 trading days)"):
    if pivots:
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Pivot (P)", f"${pivots['pivot']:,.2f}")
        pcol2.metric("Support 1 (S1)", f"${pivots['s1']:,.2f}")
        pcol3.metric("Resistance 1 (R1)", f"${pivots['r1']:,.2f}")
        st.caption(
            f"Based on 30-day High: ${pivots['high']:,.2f} · "
            f"Low: ${pivots['low']:,.2f} · Close: ${pivots['close']:,.2f}"
        )
    else:
        st.write("Not enough historical data to compute pivot points.")

st.divider()

# ============================================================================
# UI — FUNDAMENTAL ANALYSIS
# ============================================================================

st.header("Fundamental Analysis — Relative Fair Value")

peers = get_sector_peers(info, ticker_input)
peer_pe = fetch_peer_pe_ratios(peers)
target_eps = info.get("trailingEps")
sector_avg_pe = (sum(peer_pe.values()) / len(peer_pe)) if peer_pe else None
fair_value = calculate_fair_value(sector_avg_pe, target_eps)

fcol1, fcol2, fcol3 = st.columns(3)
fcol1.metric("Sector Avg P/E", f"{sector_avg_pe:.2f}" if sector_avg_pe else "N/A")
fcol2.metric(f"{ticker_input} Trailing EPS", f"${target_eps:.2f}" if target_eps is not None else "N/A")

if fair_value:
    premium_discount = (current_price - fair_value) / fair_value * 100
    label = "Premium" if premium_discount > 0 else "Discount"
    fcol3.metric(
        "Implied Fair Value",
        f"${fair_value:,.2f}",
        f"{premium_discount:+.1f}% ({label} to Fair Value)",
        delta_color="inverse",
    )
else:
    fcol3.metric("Implied Fair Value", "N/A")

with st.expander(f"Peer comparison ({', '.join(peers) if peers else 'none found'})"):
    if peer_pe:
        peer_df = pd.DataFrame(
            {"Ticker": list(peer_pe.keys()), "Trailing P/E": list(peer_pe.values())}
        )
        st.dataframe(peer_df, hide_index=True, use_container_width=True)
    else:
        st.write("Peer P/E data unavailable.")

st.divider()

# ============================================================================
# UI — QUALITATIVE ANALYSIS (LLM)
# ============================================================================

st.header("Market Narrative — Catalysts & Sentiment")

client = get_llm_client()
if client is None:
    st.warning(
        "No LLM API key detected. Set ANTHROPIC_API_KEY as an environment "
        "variable or Streamlit secret to enable catalyst and narrative analysis."
    )

move_date, move_pct = find_largest_move_date(hist, lookback_days=30)

with st.expander("📰 Catalyst: Largest Price Move (Last 30 Days)", expanded=True):
    if move_date is not None:
        st.write(f"**{move_date}** — Price moved **{move_pct:+.2f}%**")
        move_news = get_news_for_date(news, move_date)
        with st.spinner("Analyzing catalyst..."):
            catalyst_text = get_catalyst_analysis(client, ticker_input, move_date, move_pct, move_news)
        st.write(catalyst_text)
    else:
        st.write("Not enough data to identify a significant move.")

with st.expander("🗞️ Current Market Narrative (Last 48 Hours)", expanded=True):
    recent_news = get_recent_news(news, hours=48)
    with st.spinner("Synthesizing narrative..."):
        narrative_text = get_current_narrative(client, ticker_input, recent_news)
    st.write(narrative_text)
    if recent_news:
        st.caption("Source headlines:")
        st.markdown(format_headlines(recent_news))

st.divider()
st.caption("Data provided by Yahoo Finance via yfinance. Not investment advice.")
