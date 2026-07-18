"""
Automated Market Intelligence Dashboard
----------------------------------------
Takes a stock ticker and produces a technical, fundamental, and
qualitative (LLM-powered) analysis in a single Streamlit view.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API keys (two required):
    1. Finnhub (market data) — free, no credit card, 60 calls/minute.
       Get a key at: https://finnhub.io/register
       See `get_finnhub_api_key()` below for exactly where to put it.
       - Local dev:      set the FINNHUB_API_KEY environment variable
       - Streamlit Cloud: Advanced settings -> Secrets ->
                               FINNHUB_API_KEY = "..."

    2. Google Gemini (LLM narrative) — free tier, no credit card.
       Get a key at: https://aistudio.google.com/apikey
       See `get_llm_client()` below for exactly where to put it.
       - Local dev:      set the GEMINI_API_KEY environment variable
       - Streamlit Cloud: Advanced settings -> Secrets ->
                               GEMINI_API_KEY = "AIza..."
"""

import os
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types as genai_types

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Automated Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
)

LLM_MODEL = "gemini-3.1-flash-lite"

# Industry keyword -> 3-4 comparable peer tickers. Matched via substring
# search against Finnhub's `finnhubIndustry` classification string (Finnhub
# doesn't expose a fixed sector/industry enum the way Yahoo does, so a
# keyword match is more robust than an exact-string lookup table).
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


# ============================================================================
# LLM CLIENT (Google Gemini)
# ============================================================================


def get_llm_client():
    """
    Returns a google.genai.Client, or None if no key is configured.

    >>> INSERT YOUR API KEY <<<
    Get a free key (no credit card required) at https://aistudio.google.com/apikey
    Set it as the environment variable GEMINI_API_KEY, or (on Streamlit
    Community Cloud) add GEMINI_API_KEY under Advanced Settings -> Secrets.
    Never hardcode the key directly in this file.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def call_llm(client, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    if client is None:
        return "LLM analysis unavailable — no GEMINI_API_KEY configured."
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""
    except Exception as e:
        return f"LLM request failed: {e}"


# ============================================================================
# DATA FETCHING (Finnhub)
# ============================================================================


def get_finnhub_api_key():
    """
    Returns the Finnhub API key, or None if not configured.

    >>> INSERT YOUR API KEY <<<
    Get a free key (no credit card required) at https://finnhub.io/register
    Set it as the environment variable FINNHUB_API_KEY, or (on Streamlit
    Community Cloud) add FINNHUB_API_KEY under Advanced Settings -> Secrets.
    Never hardcode the key directly in this file.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["FINNHUB_API_KEY"]
        except Exception:
            api_key = None
    return api_key


def _with_retry(func, retries: int = 3, base_delay: float = 2.0):
    """
    Retries an API call with exponential backoff.

    Free-tier data APIs (Finnhub included) can occasionally return a rate
    limit or transient error. A short backoff usually clears it without the
    user having to manually retry.
    """
    last_exc = None
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                time.sleep(base_delay * (2**attempt))
    raise last_exc


def candles_to_dataframe(candles: dict) -> pd.DataFrame:
    """
    Converts a Finnhub /stock/candle response into an OHLCV DataFrame
    shaped like {Open, High, Low, Close, Volume} indexed by date — this
    matches what the technical-analysis functions below expect.
    """
    if not candles or candles.get("s") != "ok":
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "Open": candles["o"],
            "High": candles["h"],
            "Low": candles["l"],
            "Close": candles["c"],
            "Volume": candles["v"],
        },
        index=pd.to_datetime(candles["t"], unit="s"),
    )
    return df.sort_index()


# Cached for 30 minutes — Finnhub's free tier is generous (60 calls/min),
# but caching still avoids redundant work on repeat page loads.
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_data(ticker: str, api_key: str):
    """Pulls everything needed from Finnhub in one cached call."""
    client = finnhub.Client(api_key=api_key)

    def _load_core():
        p = client.company_profile2(symbol=ticker)
        q = client.quote(ticker)
        # An invalid ticker or a rate-limited request both come back as an
        # empty/zeroed response rather than raising — treat that as a
        # retryable failure instead of silently returning blank data.
        if not p or not q or not q.get("c"):
            raise RuntimeError("Empty response from Finnhub (invalid ticker or rate limit)")
        return p, q

    # Let this propagate — the caller shows the real error message, and a
    # failed call must not get cached (a silently-swallowed error here was
    # previously cached as a "successful" empty result for 30 minutes).
    profile, quote = _with_retry(_load_core)

    end = datetime.now()
    start = end - timedelta(days=180)  # comfortably covers the 50-day SMA + 30-day pivot lookback
    try:
        candles = _with_retry(
            lambda: client.stock_candles(ticker, "D", int(start.timestamp()), int(end.timestamp()))
        )
        hist = candles_to_dataframe(candles)
    except Exception:
        hist = pd.DataFrame()

    try:
        news = (
            client.company_news(
                ticker,
                _from=(end - timedelta(days=30)).strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
            )
            or []
        )
    except Exception:
        news = []

    try:
        metrics = client.company_basic_financials(ticker, "all") or {}
    except Exception:
        metrics = {}

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

    return profile, quote, hist, news, metrics, earnings


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peer_pe_ratios(peers: list, api_key: str) -> dict:
    client = finnhub.Client(api_key=api_key)
    pe_data = {}
    for peer in peers:
        try:
            metrics = _with_retry(
                lambda p=peer: client.company_basic_financials(p, "all"), retries=2, base_delay=1.5
            )
            pe = (metrics.get("metric") or {}).get("peTTM")
            if pe and pe > 0:
                pe_data[peer] = pe
        except Exception:
            continue
    return pe_data


def get_next_earnings_date(earnings: dict) -> str:
    try:
        calendar_entries = earnings.get("earningsCalendar") or []
        dates = sorted(e.get("date") for e in calendar_entries if e.get("date"))
        return dates[0] if dates else "N/A"
    except Exception:
        return "N/A"


# ============================================================================
# NEWS HELPERS
# ============================================================================


def get_news_title(item: dict) -> str:
    return item.get("headline", "")


def get_news_publisher(item: dict) -> str:
    return item.get("source", "")


def get_news_datetime(item: dict):
    ts = item.get("datetime")
    if isinstance(ts, (int, float)) and ts > 0:
        return datetime.fromtimestamp(ts)
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


def get_sector_peers(industry: str, exclude_ticker: str) -> list:
    industry_lower = (industry or "").lower()
    for keywords, peers in INDUSTRY_KEYWORD_PEERS:
        if any(kw in industry_lower for kw in keywords):
            return [p for p in peers if p.upper() != exclude_ticker.upper()][:4]
    return [p for p in DEFAULT_PEERS if p.upper() != exclude_ticker.upper()][:4]


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
    return call_llm(client, "You are a concise financial analyst.", prompt, max_tokens=200)


def get_current_narrative(client, ticker, news_items) -> str:
    headlines = format_headlines(news_items)
    prompt = (
        f"Stock: {ticker}\n\n"
        f"Headlines from the last 48 hours:\n{headlines}\n\n"
        "Synthesize the current market sentiment and narrative for this "
        "stock into a tight, 3-sentence summary."
    )
    return call_llm(client, "You are a concise financial analyst.", prompt, max_tokens=250)


# ============================================================================
# UI — SIDEBAR
# ============================================================================

st.sidebar.title("Market Intelligence")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper().strip()
st.sidebar.button("Analyze", type="primary", use_container_width=True)
st.sidebar.caption("Data: Finnhub API. Narrative: Google Gemini API.")

st.title("📈 Automated Market Intelligence Dashboard")

if not ticker_input:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

finnhub_key = get_finnhub_api_key()
if not finnhub_key:
    st.error(
        "No FINNHUB_API_KEY configured. Get a free key (no credit card "
        "required) at https://finnhub.io/register, then set it as an "
        "environment variable or Streamlit secret."
    )
    st.stop()

with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        profile, quote, hist, news, metrics, earnings = fetch_ticker_data(ticker_input, finnhub_key)
    except Exception as e:
        st.error(f"Could not fetch data for '{ticker_input}': {e}")
        st.stop()

if not quote or not quote.get("c") or hist.empty:
    st.error(
        f"No data returned for '{ticker_input}'. This usually means either the "
        "symbol is wrong, or Finnhub is rate-limiting requests — if you're sure "
        "the ticker is correct, wait a moment and click Analyze again."
    )
    st.stop()

current_price = quote.get("c")
company_name = profile.get("name", ticker_input)
previous_close = quote.get("pc", current_price)
price_change = current_price - previous_close if previous_close else 0
price_change_pct = (price_change / previous_close * 100) if previous_close else 0
next_earnings = get_next_earnings_date(earnings)

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

peers = get_sector_peers(profile.get("finnhubIndustry"), ticker_input)
peer_pe = fetch_peer_pe_ratios(peers, finnhub_key)
target_eps = (metrics.get("metric") or {}).get("epsTTM")
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
        "No LLM API key detected. Set GEMINI_API_KEY as an environment "
        "variable or Streamlit secret to enable catalyst and narrative analysis. "
        "Get a free key at https://aistudio.google.com/apikey"
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
st.caption("Data provided by Finnhub. Not investment advice.")
