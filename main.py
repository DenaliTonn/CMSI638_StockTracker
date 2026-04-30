import os
from datetime import datetime, timedelta
from fastmcp import FastMCP
from massive import RESTClient
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

mcp = FastMCP("StockAnalyzer")
client = RESTClient(os.getenv("MASSIVE_API_KEY"))

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _fetch_hourly(ticker: str, n_bars: int = 600) -> pd.DataFrame:
    """Fetch hourly OHLCV bars and return a clean DataFrame."""
    end = datetime.now()
    start = end - timedelta(days=max(60, n_bars // 16))
    aggs = client.get_aggs(
        ticker=ticker.upper(),
        multiplier=1,
        timespan="hour",
        from_=start.strftime("%Y-%m-%d"),
        to=end.strftime("%Y-%m-%d"),
    )
    if not aggs:
        raise ValueError(f"No hourly data returned for {ticker}")
    df = pd.DataFrame([{
        "open":   a.open,
        "high":   a.high,
        "low":    a.low,
        "close":  a.close,
        "volume": a.volume,
    } for a in aggs]).dropna()
    return df.reset_index(drop=True)


def _fetch_daily(ticker: str, n_bars: int = 600) -> pd.DataFrame:
    """Fetch daily OHLCV bars and return a clean DataFrame."""
    end = datetime.now()
    start = end - timedelta(days=n_bars + 30)
    aggs = client.get_aggs(
        ticker=ticker.upper(),
        multiplier=1,
        timespan="day",
        from_=start.strftime("%Y-%m-%d"),
        to=end.strftime("%Y-%m-%d"),
    )
    if not aggs:
        raise ValueError(f"No daily data returned for {ticker}")
    df = pd.DataFrame([{
        "open":   a.open,
        "high":   a.high,
        "low":    a.low,
        "close":  a.close,
        "volume": a.volume,
    } for a in aggs]).dropna()
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
#  Individual signal computation functions
# ─────────────────────────────────────────────

def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _chaikin_ad(df: pd.DataFrame) -> pd.Series:
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    ).replace(0, np.nan)
    return (clv * df["volume"]).cumsum()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def _mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    pos = rmf.where(tp > tp.shift(), 0)
    neg = rmf.where(tp < tp.shift(), 0)
    pos_sum = pos.rolling(period).sum()
    neg_sum = neg.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _bb_position(df: pd.DataFrame, period: int = 20) -> pd.Series:
    mid = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return (df["close"] - (mid - 2 * std)) / (4 * std).replace(0, np.nan)


def _compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 32 alpha signals on the full DataFrame. Returns a signals DataFrame."""
    s = pd.DataFrame(index=df.index)

    # ── Volume ──────────────────────────────────────────────────────────────
    s["vol_7h"]      = df["volume"].rolling(7).mean()
    s["vol_35h"]     = df["volume"].rolling(35).mean()
    s["vol_147h"]    = df["volume"].rolling(147).mean()
    s["vol_accel"]   = s["vol_7h"] / s["vol_35h"] - 1
    s["vol_ratio"]   = df["volume"] / s["vol_7h"]
    s["vol_norm_ret"] = (df["close"].pct_change() * df["volume"]).rolling(7).sum()

    # ── Momentum / ROC ───────────────────────────────────────────────────────
    s["momentum_7h"]   = df["close"].pct_change(7)
    s["momentum_35h"]  = df["close"].pct_change(35)
    s["momentum_147h"] = df["close"].pct_change(147)
    s["roc_7h"]        = s["momentum_7h"]
    s["roc_35h"]       = s["momentum_35h"]

    # ── RSI ──────────────────────────────────────────────────────────────────
    s["rsi_14"] = _rsi(df["close"], 14)
    s["rsi_50"] = _rsi(df["close"], 50)

    # ── MACD / PPO ───────────────────────────────────────────────────────────
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    s["macd_hist"]   = macd_line - signal_line
    s["macd_signal"] = signal_line
    s["ppo"]         = (macd_line / ema26) * 100

    # ── Volatility ────────────────────────────────────────────────────────────
    atr = _atr(df, 14)
    s["atr_norm"] = atr / df["close"]
    s["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # ── MA Distance ───────────────────────────────────────────────────────────
    s["dist_ma35"]  = df["close"] / df["close"].rolling(35).mean() - 1
    s["dist_ma147"] = df["close"] / df["close"].rolling(147).mean() - 1
    s["dist_ma441"] = df["close"] / df["close"].rolling(441).mean() - 1
    s["pct_from_high"] = df["close"] / df["high"].rolling(147).max() - 1

    # ── Candlestick ────────────────────────────────────────────────────────────
    body_top    = df[["open", "close"]].max(axis=1)
    body_bottom = df[["open", "close"]].min(axis=1)
    s["upper_wick"]   = (df["high"] - body_top) / df["close"]
    s["lower_wick"]   = (body_bottom - df["low"]) / df["close"]
    s["oc_direction"] = np.sign(df["close"] - df["open"])
    s["overnight_gap"] = df["open"] / df["close"].shift(1) - 1

    # ── Chaikin AD ────────────────────────────────────────────────────────────
    s["chaikin_ad"] = _chaikin_ad(df)

    # ── OBV signal ───────────────────────────────────────────────────────────
    obv = _obv(df)
    s["obv_signal"] = obv.pct_change(7)   # normalised 7-bar ROC of OBV

    # ── MFI ──────────────────────────────────────────────────────────────────
    s["mfi"] = _mfi(df, 14)

    # ── Amihud illiquidity ────────────────────────────────────────────────────
    s["amihud"] = (df["close"].pct_change().abs() / df["volume"].replace(0, np.nan)
                   ).rolling(21).mean()

    # ── Bollinger Band Position ───────────────────────────────────────────────
    s["bb_position"] = _bb_position(df, 20)

    # ── Squared return (lag-1) ────────────────────────────────────────────────
    s["sq_ret_lag1"] = df["close"].pct_change().shift(1) ** 2

    return s


def _classify_pattern(ic_vals: dict, horizons: list) -> str:
    vals = [ic_vals.get(h, 0) for h in horizons]
    signs = [v > 0 for v in vals]
    if all(signs):      return "always +"
    if not any(signs):  return "always -"
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    if all(d > 0 for d in diffs):  return "grows"
    if all(d < 0 for d in diffs):  return "decays"
    if vals[0] < 0 and vals[-1] > 0: return "rev→mom"
    return "mixed"


# ─────────────────────────────────────────────
#  MCP Tools
# ─────────────────────────────────────────────

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Fetches the most recent closing price for a ticker."""
    ticker = ticker.upper()
    for i in range(1, 5):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            res = client.get_daily_open_close_agg(ticker, target_date)
            if res and hasattr(res, "close"):
                return f"The closing price for {ticker} on {target_date} was ${res.close:.2f}."
        except Exception:
            continue
    return f"No data found for {ticker} in the last 4 days."


@mcp.tool()
def calculate_momentum(ticker: str, lookback_days: int = 40) -> str:
    """
    Calculates the Rate of Change (ROC) momentum factor for a given ticker.
    Positive = upward momentum; negative = downward momentum.
    """
    try:
        df = _fetch_daily(ticker, n_bars=lookback_days + 20)
        latest_close   = df["close"].iloc[-1]
        idx            = max(0, len(df) - lookback_days - 1)
        historical_close = df["close"].iloc[idx]
        roc = ((latest_close - historical_close) / historical_close) * 100
        return (
            f"Alpha Factor (Momentum): The {lookback_days}-day Rate of Change "
            f"for {ticker.upper()} is {roc:.2f}%. "
            f"(Latest: ${latest_close:.2f}, Historical: ${historical_close:.2f})"
        )
    except Exception as e:
        return f"Error calculating momentum for {ticker}: {str(e)}"


@mcp.tool()
def compute_all_indicators(ticker: str) -> str:
    """
    Computes all 32 alpha signals from the IC-by-Horizon table for the latest bar.

    Signals returned:
      Volume:     vol_7h, vol_35h, vol_147h, vol_accel, vol_ratio, vol_norm_ret
      Momentum:   momentum_7h, momentum_35h, momentum_147h, roc_7h, roc_35h
      RSI:        rsi_14, rsi_50
      MACD:       macd_hist, macd_signal, ppo
      Volatility: atr_norm, hl_range
      MA Dist:    dist_ma35, dist_ma147, dist_ma441, pct_from_high
      Candle:     upper_wick, lower_wick, oc_direction, overnight_gap
      Other:      chaikin_ad, obv_signal, mfi, amihud, bb_position, sq_ret_lag1
    """
    try:
        df = _fetch_hourly(ticker, n_bars=600)
        signals = _compute_signals(df)
        latest = signals.iloc[-1].dropna()

        lines = [f"── {ticker.upper()} Latest Signal Snapshot ──"]
        groups = {
            "Volume":     ["vol_7h", "vol_35h", "vol_147h", "vol_accel", "vol_ratio", "vol_norm_ret"],
            "Momentum":   ["momentum_7h", "momentum_35h", "momentum_147h", "roc_7h", "roc_35h"],
            "RSI":        ["rsi_14", "rsi_50"],
            "MACD/PPO":   ["macd_hist", "macd_signal", "ppo"],
            "Volatility": ["atr_norm", "hl_range"],
            "MA Distance":["dist_ma35", "dist_ma147", "dist_ma441", "pct_from_high"],
            "Candle":     ["upper_wick", "lower_wick", "oc_direction", "overnight_gap"],
            "Other":      ["chaikin_ad", "obv_signal", "mfi", "amihud", "bb_position", "sq_ret_lag1"],
        }
        for group, cols in groups.items():
            lines.append(f"\n[{group}]")
            for col in cols:
                val = latest.get(col, float("nan"))
                lines.append(f"  {col:<18} {val:+.4f}" if not np.isnan(val) else f"  {col:<18} N/A")

        return "\n".join(lines)
    except Exception as e:
        return f"Error computing indicators for {ticker}: {str(e)}"


@mcp.tool()
def compute_ic_table(ticker: str, min_rows: int = 300) -> str:
    """
    Computes the Information Coefficient (IC) table for a ticker — identical
    in structure to the GOOGL Hourly – IC by Horizon report.

    For each of the 32 signals, reports Spearman rank correlation vs.
    forward returns at 1h, 4h, 8h, and 24h horizons, the best horizon,
    and the IC decay pattern (grows / decays / always+ / always- / rev→mom).
    """
    try:
        df = _fetch_hourly(ticker, n_bars=600)
        if len(df) < min_rows:
            return f"Insufficient data for {ticker}: only {len(df)} hourly bars available (need {min_rows}+)."

        signals = _compute_signals(df)
        HORIZONS = [1, 4, 8, 24]

        # Forward returns at each horizon
        fwd = {h: df["close"].shift(-h) / df["close"] - 1 for h in HORIZONS}

        rows = []
        for sig in signals.columns:
            row = {"signal": sig}
            ic_vals = {}
            for h in HORIZONS:
                mask = signals[sig].notna() & fwd[h].notna()
                if mask.sum() < 30:
                    ic_vals[h] = float("nan")
                    continue
                ic, _ = spearmanr(signals[sig][mask], fwd[h][mask])
                ic_vals[h] = round(ic, 3)

            row.update({f"{h}h": ic_vals[h] for h in HORIZONS})

            valid_ics = {h: v for h, v in ic_vals.items() if not np.isnan(v)}
            if valid_ics:
                best_h = max(valid_ics, key=lambda h: abs(valid_ics[h]))
                row["best"]    = f"{best_h}h"
                row["pattern"] = _classify_pattern(ic_vals, HORIZONS)
            else:
                row["best"]    = "N/A"
                row["pattern"] = "N/A"

            rows.append(row)

        # Format as aligned table
        header = (
            f"\n── {ticker.upper()} HOURLY – IC BY HORIZON ──\n"
            f"{'Signal':<18} {'1h':>8} {'4h':>8} {'8h':>8} {'24h':>8}  {'Best':<6}  Pattern\n"
            + "─" * 72
        )
        lines = [header]
        for r in sorted(rows, key=lambda x: abs(x.get("24h") or 0), reverse=True):
            def fmt(v):
                return f"{v:+.3f}" if isinstance(v, float) and not np.isnan(v) else "  N/A "
            lines.append(
                f"{r['signal']:<18} {fmt(r['1h']):>8} {fmt(r['4h']):>8} "
                f"{fmt(r['8h']):>8} {fmt(r['24h']):>8}  {r['best']:<6}  {r['pattern']}"
            )
        return "\n".join(lines)

    except Exception as e:
        return f"Error computing IC table for {ticker}: {str(e)}"


@mcp.tool()
def get_signal_detail(ticker: str, signal_name: str) -> str:
    """
    Returns the last 10 values of a single named signal for a ticker,
    along with a summary (mean, std, latest).

    Valid signal names match those in compute_all_indicators / compute_ic_table.
    """
    try:
        df = _fetch_hourly(ticker, n_bars=600)
        signals = _compute_signals(df)
        if signal_name not in signals.columns:
            valid = ", ".join(signals.columns.tolist())
            return f"Unknown signal '{signal_name}'. Valid signals: {valid}"

        series = signals[signal_name].dropna()
        last10 = series.tail(10).tolist()
        summary = (
            f"Signal: {signal_name} | Ticker: {ticker.upper()}\n"
            f"  Latest : {series.iloc[-1]:+.4f}\n"
            f"  Mean   : {series.mean():+.4f}\n"
            f"  Std    : {series.std():.4f}\n"
            f"  Min    : {series.min():+.4f}\n"
            f"  Max    : {series.max():+.4f}\n"
            f"  Last 10: {[round(v, 4) for v in last10]}"
        )
        return summary
    except Exception as e:
        return f"Error fetching signal detail for {ticker}/{signal_name}: {str(e)}"


# ─────────────────────────────────────────────
#  Commented-out: NLP Sentiment (MarketAux)
# ─────────────────────────────────────────────

# @mcp.tool()
# async def get_sentiment_analysis(ticker: str) -> str:
#     """Real NLP Sentiment Agent via MarketAux."""
#     import httpx
#     api_token = os.getenv("MARKETAUX_API_KEY")
#     ticker = ticker.upper()
#     url = "https://api.marketaux.com/v1/news/all"
#     params = {"symbols": ticker, "filter_entities": "true",
#               "language": "en", "api_token": api_token}
#     async with httpx.AsyncClient() as c:
#         response = await c.get(url, params=params)
#         if response.status_code != 200:
#             return f"Error: Unable to fetch news (Status {response.status_code})"
#         results = response.json().get("data", [])
#         if not results:
#             return f"No recent news found for {ticker}."
#         top_news = results[0]
#         sentiment_score = 0.0
#         for entity in top_news.get("entities", []):
#             if entity.get("symbol") == ticker:
#                 sentiment_score = entity.get("sentiment_score", 0.0)
#         return (f"NLP Sentiment Factor: {ticker} score={sentiment_score:.2f}. "
#                 f"Top Headline: {top_news['title']}. URL: {top_news['url']}")


if __name__ == "__main__":
    mcp.run()