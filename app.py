"""
Automated Market Intelligence Dashboard
----------------------------------------
Takes a stock ticker and produces a technical and fundamental analysis
(valuation multiples, analyst price targets, and a DCF intrinsic value)
in a single Streamlit view.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

API keys (two required):
    1. Finnhub (quotes, fundamentals, peers, earnings) — free, no credit
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

    Analyst price targets use yfinance (Yahoo Finance) — no key required.
    Finnhub's price-target endpoint is premium, so this is the free source.
"""

import os
import statistics
import time
from datetime import datetime, timedelta

import finnhub
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ============================================================================
# CONFIG
# ============================================================================

st.set_page_config(
    page_title="Automated Market Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
)

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

# Fundamental-analysis tuning knobs.
PEER_LIMIT = 20          # top-N peers by market cap to include in the median
PE_OUTLIER_MAX = 200.0   # discard P/E above this as an extreme outlier


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
    """Pulls quote/profile/fundamentals/earnings from Finnhub in one cached call."""
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

    return profile, quote, metrics, earnings


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


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_analyst_targets(ticker: str, finnhub_key: str) -> dict:
    """
    Analyst price-target consensus (low / mean / high). Tries Finnhub first
    (in case the plan includes it), then falls back to yfinance's free
    `.info` fields. Returns None if neither source has targets.
    """
    # Finnhub /stock/price-target — premium on most plans, but try anyway.
    try:
        client = finnhub.Client(api_key=finnhub_key)
        pt = _with_retry(lambda: client.price_target(ticker), retries=1, base_delay=1.0) or {}
        if pt.get("targetMean"):
            return {
                "low": pt.get("targetLow"),
                "mean": pt.get("targetMean"),
                "high": pt.get("targetHigh"),
                "count": None,
                "source": "Finnhub",
            }
    except Exception:
        pass
    # yfinance free fallback (Yahoo Finance).
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
        calendar_entries = earnings.get("earningsCalendar") or []
        dates = sorted(e.get("date") for e in calendar_entries if e.get("date"))
        return dates[0] if dates else "N/A"
    except Exception:
        return "N/A"


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


def peg_ratio(pe, growth_pct):
    """PEG = P/E ÷ annual earnings growth (in whole percent). Under 1.0 is the
    classic 'cheap relative to growth' threshold for growth stocks."""
    if not pe or pe <= 0 or not growth_pct or growth_pct <= 0:
        return None
    return pe / growth_pct


def dcf_intrinsic_value(base_fcf_ps, growth, discount_rate, terminal_growth, years):
    """
    Two-stage DCF on a per-share basis (the practical form of
    Intrinsic Value = Σ FCF_t / (1 + r)^t): project free cash flow per share
    for `years` at `growth`, discount each year at `discount_rate` (WACC),
    add a Gordon-growth terminal value, and sum the present values.

        intrinsic = Σ_{t=1..N} FCF_t/(1+r)^t  +  TV/(1+r)^N
        where TV = FCF_N·(1+g_term) / (r − g_term)

    Per-share, no net-debt adjustment (a deliberate simplification).
    Returns None if inputs are unusable.
    """
    if not base_fcf_ps or base_fcf_ps <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None  # terminal value diverges when r <= g_terminal
    pv_sum = 0.0
    fcf = base_fcf_ps
    for t in range(1, years + 1):
        fcf = base_fcf_ps * ((1 + growth) ** t)
        pv_sum += fcf / ((1 + discount_rate) ** t)
    terminal_value = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_sum += terminal_value / ((1 + discount_rate) ** years)
    return pv_sum


# ============================================================================
# UI — SIDEBAR
# ============================================================================

st.sidebar.title("Market Intelligence")
ticker_input = st.sidebar.text_input("Enter Stock Ticker", value="NVDA").upper().strip()
st.sidebar.button("Analyze", type="primary", use_container_width=True)
st.sidebar.caption("Data: Finnhub · Twelve Data · Yahoo Finance (analyst targets).")

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
        profile, quote, metrics, earnings = fetch_ticker_data(ticker_input, finnhub_key)
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

st.header("Fundamental Analysis — Valuation")

# ---- Peer group + outlier-resistant median sector P/E ----
peers = fetch_peer_group(ticker_input, profile.get("finnhubIndustry"), finnhub_key)
peer_rows = fetch_peer_metrics(peers, finnhub_key)
sector = compute_sector_multiples(peer_rows)
sector_median_pe = sector["median_pe"]

# ---- Company fundamentals from Finnhub ----
trailing_pe = metric_value(metrics, "peTTM")
fcf_per_share = metric_value(
    metrics, "freeCashFlowPerShareTTM", "freeCashFlowPerShareAnnual", "cashFlowPerShareTTM"
)
auto_growth = normalize_growth(
    metric_value(metrics, "epsGrowth5Y", "epsGrowth3Y", "epsGrowthTTMYoy", "revenueGrowth5Y")
)
default_growth_pct = max(1, min(int(round((auto_growth if auto_growth is not None else 0.10) * 100)), 60))

# ---- Adjustable assumptions (auto-seeded where free data allows) ----
with st.expander("⚙️ Valuation assumptions (adjustable)", expanded=False):
    st.caption(
        "Growth and DCF inputs auto-seed from available data where possible "
        "(analyst consensus estimates are premium-only on the free tiers), and "
        "are yours to tune."
    )
    ac1, ac2 = st.columns(2)
    with ac1:
        growth_pct = st.slider("Expected annual EPS growth (for PEG)", 1, 60, default_growth_pct, 1,
                               format="%d%%", key="peg_growth")
        dcf_g_pct = st.slider("DCF · FCF growth (stage 1)", 0, 30, min(default_growth_pct, 30), 1,
                              format="%d%%", key="dcf_g")
        dcf_years = st.slider("DCF · projection years", 5, 15, 10, 1, key="dcf_years")
    with ac2:
        dcf_base_fcf = st.number_input(
            "DCF · base FCF per share ($)",
            value=float(round(fcf_per_share, 2)) if isinstance(fcf_per_share, (int, float)) else 0.0,
            step=0.10, key="dcf_fcf",
            help="Auto-seeded from Finnhub when available; adjust to your own estimate.",
        )
        dcf_r_pct = st.slider("DCF · discount rate (WACC)", 4, 15, 9, 1, format="%d%%", key="dcf_r")
        dcf_tg_pct = st.slider("DCF · terminal growth", 0, 5, 3, 1, format="%d%%", key="dcf_tg")

# ---- Valuation multiples: P/E, PEG, Sector Median P/E ----
peg = peg_ratio(trailing_pe, growth_pct)
m1, m2, m3 = st.columns(3)
m1.metric(
    "P/E Ratio (trailing)", f"{trailing_pe:.1f}x" if trailing_pe else "N/A",
    help="Price ÷ EPS — what you pay per $1 of trailing profit. A lower P/E can "
         "signal undervaluation; a higher one reflects strong growth expectations.",
)
if peg is not None:
    m2.metric(
        "PEG Ratio", f"{peg:.2f}",
        "Below 1.0 — cheap vs growth" if peg < 1 else "Above 1.0 — rich vs growth",
        delta_color="off",
        help=f"P/E ÷ growth ({growth_pct}%). Under 1.0 is the classic bargain "
             "threshold relative to a growth stock's earnings growth.",
    )
else:
    m2.metric("PEG Ratio", "N/A", help="Needs a positive P/E and a positive growth rate.")
m3.metric(
    "Sector Median P/E", f"{sector_median_pe:.1f}x" if sector_median_pe else "N/A",
    f"{sector['count']} peers" if sector["count"] else None, delta_color="off",
    help="Median (not mean) of cleansed sub-industry peers — resistant to mega-cap outliers.",
)

# ---- Analyst price targets: MIN / AVG / MAX ----
st.subheader("Analyst Price Targets (next 12 months)")
targets = fetch_analyst_targets(ticker_input, finnhub_key)
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
        st.bar_chart(chart_df, color="#4C9BE8")
else:
    st.info(
        "No analyst price targets available for this ticker. Finnhub's price-target "
        "endpoint is premium, and the Yahoo Finance fallback returned none (it may be "
        "rate-limited or the ticker isn't covered)."
    )

# ---- Intrinsic value: Discounted Cash Flow ----
st.subheader("Intrinsic Value — Discounted Cash Flow (DCF)")
intrinsic = dcf_intrinsic_value(dcf_base_fcf, dcf_g_pct / 100, dcf_r_pct / 100, dcf_tg_pct / 100, dcf_years)
if intrinsic and intrinsic > 0:
    gap = (intrinsic - current_price) / current_price * 100
    verdict = "Undervalued" if intrinsic > current_price else "Overvalued"
    st.metric("DCF Intrinsic Value / share", f"${intrinsic:,.2f}",
              f"{gap:+.1f}% vs price ({verdict})")
    st.caption(
        f"2-stage DCF: {dcf_years}y at {dcf_g_pct}% FCF growth, {dcf_r_pct}% discount "
        f"(WACC), {dcf_tg_pct}% terminal growth, base FCF/share ${dcf_base_fcf:.2f}. "
        "Per-share, no net-debt adjustment (simplified)."
    )
else:
    st.metric("DCF Intrinsic Value / share", "N/A")
    if dcf_r_pct <= dcf_tg_pct:
        st.caption("Discount rate must exceed terminal growth for the model to converge.")
    else:
        st.caption("Set a positive base FCF per share in the assumptions above to run the DCF.")

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
st.caption("Data: Finnhub, Twelve Data, and Yahoo Finance. Not investment advice.")
