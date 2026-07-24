"""
Automated Market Intelligence Dashboard
----------------------------------------
Takes a stock ticker and produces a technical, fundamental, and
qualitative (LLM-powered) analysis in a single Streamlit view.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API keys (three required):
    1. Finnhub (quotes, fundamentals, news, earnings) — free, no credit
       card, 60 calls/minute.
       Get a key at: https://finnhub.io/register
       See `get_finnhub_api_key()` below for exactly where to put it.
       - Local dev:      set the FINNHUB_API_KEY environment variable
       - Streamlit Cloud: Advanced settings -> Secrets ->
                               FINNHUB_API_KEY = "..."

    2. Twelve Data (historical daily prices) — free, no credit card,
       800 calls/day. Finnhub's historical-price endpoint is paid-only,
       so this covers the SMA/pivot-point calculations instead.
       Get a key at: https://twelvedata.com/pricing (Basic/free plan)
       See `get_twelvedata_api_key()` below for exactly where to put it.
       - Local dev:      set the TWELVEDATA_API_KEY environment variable
       - Streamlit Cloud: Advanced settings -> Secrets ->
                               TWELVEDATA_API_KEY = "..."

    3. Google Gemini (LLM narrative) — free tier, no credit card.
       Get a key at: https://aistudio.google.com/apikey
       See `get_llm_client()` below for exactly where to put it.
       - Local dev:      set the GEMINI_API_KEY environment variable
       - Streamlit Cloud: Advanced settings -> Secrets ->
                               GEMINI_API_KEY = "AIza..."
"""

import os
import statistics
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import requests
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

# Fundamental-analysis tuning knobs (see the framework in the README/commit).
PEER_LIMIT = 20          # top-N peers by market cap to include in the median
PE_OUTLIER_MAX = 200.0   # discard P/E above this as an extreme outlier
HIST_PE_YEARS = 5        # window for the historical-average-P/E anchor multiple


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


def get_twelvedata_api_key():
    """
    Returns the Twelve Data API key, or None if not configured.

    >>> INSERT YOUR API KEY <<<
    Get a free key (no credit card required, Basic plan) at
    https://twelvedata.com/pricing
    Set it as the environment variable TWELVEDATA_API_KEY, or (on Streamlit
    Community Cloud) add TWELVEDATA_API_KEY under Advanced Settings -> Secrets.
    Never hardcode the key directly in this file.
    """
    api_key = os.environ.get("TWELVEDATA_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["TWELVEDATA_API_KEY"]
        except Exception:
            api_key = None
    return api_key


# Cached for 30 minutes — Twelve Data's free tier is 800 calls/day, so
# caching keeps repeat page loads from eating into that budget.
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price_history(ticker: str, api_key: str) -> pd.DataFrame:
    """
    Daily OHLCV history from Twelve Data, shaped like {Open, High, Low,
    Close, Volume} indexed by date — this is what Finnhub's historical
    /stock/candle endpoint would have returned, but that endpoint is
    paid-only on Finnhub's free tier, so this covers it instead.
    """

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


# Cached for 30 minutes — Finnhub's free tier is generous (60 calls/min),
# but caching still avoids redundant work on repeat page loads.
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ticker_data(ticker: str, api_key: str):
    """Pulls quote/fundamentals/news/earnings from Finnhub in one cached call."""
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

    return profile, quote, news, metrics, earnings


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_peer_group(ticker: str, industry: str, api_key: str) -> list:
    """
    Dynamic sub-industry peer group from Finnhub's /stock/peers endpoint
    (free tier, defaults to sub-industry grouping — the GICS-style peer
    definition the framework's Part 1 calls for). Falls back to the static
    keyword map if the endpoint returns nothing.
    """
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
def fetch_peer_metrics(peers: list, api_key: str) -> list:
    """
    Per-peer trailing P/E, P/S, and market cap. (Forward P/E and EV/Sales,
    which the framework ideally wants here, require analyst estimates that
    are premium-only on Finnhub's free tier — trailing P/E is the free
    proxy, and P/S backs the unprofitable-company valuation fallback.)
    """
    client = finnhub.Client(api_key=api_key)
    rows = []
    for peer in peers:
        try:
            data = _with_retry(
                lambda p=peer: client.company_basic_financials(p, "all"), retries=2, base_delay=1.5
            )
            metric = data.get("metric") or {}
            rows.append(
                {
                    "ticker": peer,
                    "pe": metric.get("peTTM"),
                    "ps": metric.get("psTTM"),
                    "market_cap": metric.get("marketCapitalization"),
                }
            )
        except Exception:
            continue
    return rows


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


def get_sector_peers_static(industry: str, exclude_ticker: str) -> list:
    """Fallback peer group when Finnhub's /stock/peers returns nothing."""
    industry_lower = (industry or "").lower()
    for keywords, peers in INDUSTRY_KEYWORD_PEERS:
        if any(kw in industry_lower for kw in keywords):
            return [p for p in peers if p.upper() != exclude_ticker.upper()]
    return [p for p in DEFAULT_PEERS if p.upper() != exclude_ticker.upper()]


def metric_value(metrics: dict, *names):
    """First present numeric value among candidate metric keys (Finnhub's
    field names vary), else None."""
    m = (metrics or {}).get("metric") or {}
    for n in names:
        v = m.get(n)
        if isinstance(v, (int, float)):
            return v
    return None


def normalize_growth(v):
    """Finnhub growth fields are sometimes a percent (15.3) and sometimes a
    decimal (0.153). Normalize to a decimal fraction."""
    if not isinstance(v, (int, float)):
        return None
    return v / 100 if abs(v) > 3 else v


def compute_sector_multiples(peer_rows: list) -> dict:
    """
    Part 1 — sector averages via the MEDIAN (mega-cap outliers skew a mean),
    after cleansing: drop negative/zero P/E (unprofitable) and P/E above the
    outlier cap, then keep the top-N valid peers by market cap.
    """
    valid = [
        r for r in peer_rows
        if isinstance(r.get("pe"), (int, float)) and 0 < r["pe"] <= PE_OUTLIER_MAX
    ]
    if any(r.get("market_cap") for r in valid):
        valid = sorted(valid, key=lambda r: r.get("market_cap") or 0, reverse=True)[:PEER_LIMIT]
    pes = [r["pe"] for r in valid]
    pss = [r["ps"] for r in valid if isinstance(r.get("ps"), (int, float)) and r["ps"] > 0]
    return {
        "median_pe": statistics.median(pes) if pes else None,
        "mean_pe": statistics.fmean(pes) if pes else None,
        "median_ps": statistics.median(pss) if pss else None,
        "valid_peers": valid,
        "count": len(pes),
    }


def get_historical_avg_pe(metrics: dict, years: int = HIST_PE_YEARS):
    """
    5-year average trailing P/E from Finnhub's annual `series` (the anchor
    multiple M_historical in Part 2). Returns None when the series isn't
    available for this ticker/tier.
    """
    try:
        annual = ((metrics or {}).get("series") or {}).get("annual") or {}
        pe_series = annual.get("pe") or []
        values = [
            pt.get("v")
            for pt in pe_series
            if isinstance(pt.get("v"), (int, float)) and 0 < pt["v"] <= PE_OUTLIER_MAX
        ]
        values = values[:years]  # Finnhub returns the series newest-first
        return statistics.fmean(values) if values else None
    except Exception:
        return None


def estimate_forward_eps(eps_ttm, growth):
    """NTM EPS proxy: trailing EPS grown one year (true consensus NTM EPS is
    premium-only, so this is derived from EPS_ttm and the growth assumption)."""
    if eps_ttm is None or growth is None:
        return None
    return eps_ttm * (1 + growth)


def historical_multiple_fair_value(eps_fwd, m_historical):
    """Part 2 — P_fair = EPS_fwd × M_historical."""
    if eps_fwd is None or m_historical is None:
        return None
    return eps_fwd * m_historical


def ps_fair_value(sector_median_ps, sales_per_share):
    """Unprofitable-company fallback (EPS < 0): value on Price/Sales instead."""
    if not sector_median_ps or not sales_per_share:
        return None
    return sector_median_ps * sales_per_share


def scenario_target(eps_ttm, growth, multiple, t: int = 1):
    """Part 3 — P_target = EPS_ttm × (1 + g)^t × M_target."""
    if eps_ttm is None or multiple is None or growth is None:
        return None
    return eps_ttm * ((1 + growth) ** t) * multiple


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
st.sidebar.caption("Data: Finnhub + Twelve Data. Narrative: Google Gemini API.")

st.title("📈 Automated Market Intelligence Dashboard")

if not ticker_input:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

finnhub_key = get_finnhub_api_key()
twelvedata_key = get_twelvedata_api_key()
missing_keys = []
if not finnhub_key:
    missing_keys.append("FINNHUB_API_KEY (get one at https://finnhub.io/register)")
if not twelvedata_key:
    missing_keys.append("TWELVEDATA_API_KEY (get one at https://twelvedata.com/pricing)")
if missing_keys:
    st.error(
        "Missing required API key(s):\n\n"
        + "\n".join(f"- {k}" for k in missing_keys)
        + "\n\nSet these as environment variables or Streamlit secrets."
    )
    st.stop()

with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        profile, quote, news, metrics, earnings = fetch_ticker_data(ticker_input, finnhub_key)
        hist = fetch_price_history(ticker_input, twelvedata_key)
    except Exception as e:
        st.error(f"Could not fetch data for '{ticker_input}': {e}")
        st.stop()

if not quote or not quote.get("c") or hist.empty:
    st.error(
        f"No data returned for '{ticker_input}'. This usually means either the "
        "symbol is wrong, or Finnhub/Twelve Data is rate-limiting requests — "
        "if you're sure the ticker is correct, wait a moment and click Analyze again."
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

st.header("Fundamental Analysis — Relative Valuation & Scenarios")

# ---- Part 1: peer group + median sector multiples ----
peers = fetch_peer_group(ticker_input, profile.get("finnhubIndustry"), finnhub_key)
peer_rows = fetch_peer_metrics(peers, finnhub_key)
sector = compute_sector_multiples(peer_rows)
sector_median_pe = sector["median_pe"]

# ---- Company fundamentals from Finnhub ----
trailing_eps = metric_value(metrics, "epsTTM")
trailing_pe = metric_value(metrics, "peTTM")
sales_per_share = metric_value(metrics, "salesPerShareTTM", "revenuePerShareTTM")
hist_avg_pe = get_historical_avg_pe(metrics)
auto_growth = normalize_growth(
    metric_value(metrics, "epsGrowth5Y", "epsGrowthTTMYoy", "revenueGrowth5Y", "revenueGrowthTTMYoy")
)
is_unprofitable = trailing_eps is not None and trailing_eps < 0

# The anchor multiple M (Part 2/3): prefer the stock's own 5-year historical
# average P/E, else the sector median, else its current trailing P/E.
default_anchor = hist_avg_pe or sector_median_pe or trailing_pe or 20.0
anchor_source = (
    "5-year historical avg P/E" if hist_avg_pe
    else "sector median P/E" if sector_median_pe
    else "current trailing P/E" if trailing_pe
    else "default (20x)"
)
default_base_g = auto_growth if auto_growth is not None else 0.10
default_base_g = max(0.0, min(default_base_g, 0.50))  # clamp the auto-seed to a sane slider range

# ---- Adjustable assumptions (premium consensus data isn't free, so these
#      auto-seed from available data and remain user-tunable) ----
with st.expander("⚙️ Valuation assumptions (adjustable)", expanded=False):
    st.caption(
        f"Anchor multiple auto-seeded from **{anchor_source}**. "
        "Forward-looking consensus estimates are premium-only on the free data "
        "tiers, so these inputs are yours to tune."
    )
    # Sliders operate in whole-percent units for clear labels; the math
    # below converts each back to a decimal fraction (÷100).
    ac1, ac2 = st.columns(2)
    with ac1:
        base_g_pct = st.slider("Base-case EPS growth (g)", 0, 50, int(round(default_base_g * 100)),
                               1, format="%d%%", key="base_g")
        bear_g_pct = st.slider("Bear-case EPS growth", 0, 20, 5, 1, format="%d%%", key="bear_g")
        bull_g_pct = st.slider("Bull-case EPS growth", 0, 60,
                               int(round(min(default_base_g + 0.10, 0.60) * 100)), 1,
                               format="%d%%", key="bull_g")
    with ac2:
        anchor_m = st.number_input("Anchor multiple (M)", 1.0, 200.0, float(round(default_anchor, 1)),
                                   0.5, key="anchor_m")
        bear_comp_pct = st.slider("Bear multiple compression", 0, 50, 25, 5,
                                  format="%d%%", key="bear_comp")
        bull_exp_pct = st.slider("Bull multiple expansion", 0, 50, 18, 1,
                                 format="%d%%", key="bull_exp")

base_g, bear_g, bull_g = base_g_pct / 100, bear_g_pct / 100, bull_g_pct / 100
bear_compression, bull_expansion = bear_comp_pct / 100, bull_exp_pct / 100

base_m = anchor_m
bear_m = anchor_m * (1 - bear_compression)
bull_m = anchor_m * (1 + bull_expansion)

forward_eps = estimate_forward_eps(trailing_eps, base_g)
forward_pe = (current_price / forward_eps) if forward_eps and forward_eps > 0 else None

# ---- Part 2: historical-multiple fair value (P/S fallback if unprofitable) ----
if is_unprofitable:
    fair_value = ps_fair_value(sector["median_ps"], sales_per_share)
    fair_basis = "Price/Sales model (EPS < 0 — unprofitable)"
else:
    fair_value = historical_multiple_fair_value(forward_eps, base_m)
    fair_basis = f"Forward EPS × anchor multiple ({anchor_source})"

# ---- Headline multiples ----
f1, f2, f3 = st.columns(3)
f1.metric("Trailing P/E", f"{trailing_pe:.1f}x" if trailing_pe else "N/A")
f2.metric("Est. Forward P/E", f"{forward_pe:.1f}x" if forward_pe else "N/A",
          help="Price ÷ (trailing EPS grown by base-case g). True consensus "
               "forward P/E is premium-only on the free data tiers.")
f3.metric(
    "Sector Median P/E",
    f"{sector_median_pe:.1f}x" if sector_median_pe else "N/A",
    f"{sector['count']} peers" if sector["count"] else None,
    delta_color="off",
    help="Median (not mean) of cleansed sub-industry peers — resistant to "
         "mega-cap outliers.",
)

# ---- Fair value vs current price ----
if fair_value and fair_value > 0:
    premium_discount = (current_price - fair_value) / fair_value * 100
    label = "Premium" if premium_discount > 0 else "Discount"
    st.metric(
        "Implied Fair Value",
        f"${fair_value:,.2f}",
        f"{premium_discount:+.1f}% ({label} to fair value)",
        delta_color="inverse",
    )
    st.caption(f"Basis: {fair_basis}")
else:
    st.metric("Implied Fair Value", "N/A")
    st.caption(
        "Fair value unavailable — the underlying data "
        f"({'sales-per-share / sector P/S' if is_unprofitable else 'EPS / anchor multiple'}) "
        "wasn't returned for this ticker."
    )

# ---- Part 3: Bear / Base / Bull scenarios ----
st.subheader("12-Month Price Targets — Bear / Base / Bull")
scenarios = [
    ("🐻 Bear", bear_g, bear_m, scenario_target(trailing_eps, bear_g, bear_m)),
    ("⚖️ Base", base_g, base_m, scenario_target(trailing_eps, base_g, base_m)),
    ("🐂 Bull", bull_g, bull_m, scenario_target(trailing_eps, bull_g, bull_m)),
]

scol = st.columns(3)
for col, (name, g, m, target) in zip(scol, scenarios):
    if target and target > 0:
        upside = (target - current_price) / current_price * 100
        col.metric(name, f"${target:,.2f}", f"{upside:+.1f}% vs current")
        col.caption(f"{g * 100:.0f}% growth · {m:.1f}x multiple")
    else:
        col.metric(name, "N/A")
        col.caption("Needs positive trailing EPS")

# Visualize the three targets against the current price.
targets = {name.split()[-1]: t for name, _, _, t in scenarios if t and t > 0}
if targets:
    chart_df = pd.DataFrame(
        {"Price ($)": {**targets, "Current": current_price}}
    ).reindex(["Bear", "Current", "Base", "Bull"]).dropna()
    st.bar_chart(chart_df, color="#4C9BE8")

# ---- Peer detail ----
with st.expander(f"Peer group ({', '.join(peers) if peers else 'none found'})"):
    valid_peers = sector["valid_peers"]
    if valid_peers:
        peer_df = pd.DataFrame(
            [{"Ticker": r["ticker"], "Trailing P/E": round(r["pe"], 1),
              "P/S": round(r["ps"], 1) if isinstance(r.get("ps"), (int, float)) else None}
             for r in valid_peers]
        )
        st.dataframe(peer_df, hide_index=True, use_container_width=True)
        if sector["mean_pe"] and sector_median_pe:
            st.caption(
                f"Median P/E **{sector_median_pe:.1f}x** vs mean **{sector['mean_pe']:.1f}x** — "
                "the median is used precisely because the mean gets pulled by outliers. "
                "Negative and >200 P/E peers were filtered out."
            )
    else:
        st.write("No valid peer P/E data returned.")

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
st.caption("Data provided by Finnhub and Twelve Data. Not investment advice.")
