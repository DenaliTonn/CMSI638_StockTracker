"""
web_server.py — AlphaDeskQ FastAPI Server
==========================================
Wraps main.py's computation functions into REST endpoints consumed by alphadesq.html.
Run with:  uv run uvicorn web_server:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx

load_dotenv()

# ── Import computation helpers from main.py ────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from main import (
    _fetch_hourly,
    _fetch_daily,
    _compute_signals,
    _classify_pattern,
    _rsi,
    _atr,
    _chaikin_ad,
    _obv,
    _mfi,
    _bb_position,
)

from massive import RESTClient
from scipy.stats import spearmanr

client = RESTClient(os.getenv("MASSIVE_API_KEY"))

HORIZONS = [1, 4, 8, 24]
SIGNAL_GROUPS = {
    "Volume":      ["vol_7h", "vol_35h", "vol_147h", "vol_accel", "vol_ratio", "vol_norm_ret"],
    "Momentum":    ["momentum_7h", "momentum_35h", "momentum_147h", "roc_7h", "roc_35h"],
    "RSI":         ["rsi_14", "rsi_50"],
    "MACD/PPO":    ["macd_hist", "macd_signal", "ppo"],
    "Volatility":  ["atr_norm", "hl_range"],
    "MA Distance": ["dist_ma35", "dist_ma147", "dist_ma441", "pct_from_high"],
    "Candle":      ["upper_wick", "lower_wick", "oc_direction", "overnight_gap"],
    "Other":       ["chaikin_ad", "obv_signal", "mfi", "amihud", "bb_position", "sq_ret_lag1"],
}

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="AlphaDeskQ", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    ticker: str
    horizon: str = "4h"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatPayload(BaseModel):
    system: str
    messages: list[ChatMessage]


# ── Helpers ────────────────────────────────────────────────────────────────
def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _compute_ic_rows(df: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    fwd = {h: df["close"].shift(-h) / df["close"] - 1 for h in HORIZONS}
    rows: list[dict] = []

    for sig in signals.columns:
        ic_vals: dict[int, float] = {}
        for h in HORIZONS:
            mask = signals[sig].notna() & fwd[h].notna()
            if mask.sum() < 30:
                ic_vals[h] = float("nan")
                continue
            ic, _ = spearmanr(signals[sig][mask], fwd[h][mask])
            ic_vals[h] = round(float(ic), 3)

        valid_ics = {h: v for h, v in ic_vals.items() if not np.isnan(v)}
        best_h = max(valid_ics, key=lambda h: abs(valid_ics[h])) if valid_ics else None
        pattern = _classify_pattern(ic_vals, HORIZONS) if valid_ics else "N/A"

        rows.append({
            "signal":  sig,
            "ic_1h":   _safe_float(ic_vals.get(1)),
            "ic_4h":   _safe_float(ic_vals.get(4)),
            "ic_8h":   _safe_float(ic_vals.get(8)),
            "ic_24h":  _safe_float(ic_vals.get(24)),
            "best":    f"{best_h}h" if best_h else "N/A",
            "pattern": pattern,
        })

    rows.sort(key=lambda r: abs(r["ic_24h"] or 0), reverse=True)
    return rows


def _compute_predictions(df: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    fwd = {h: df["close"].shift(-h) / df["close"] - 1 for h in HORIZONS}
    latest_signals = signals.iloc[-1]
    preds = []

    for h in HORIZONS:
        signal_ics: list[tuple[str, float]] = []
        for sig in signals.columns:
            mask = signals[sig].notna() & fwd[h].notna()
            if mask.sum() < 30:
                continue
            ic, _ = spearmanr(signals[sig][mask], fwd[h][mask])
            if not np.isnan(ic):
                signal_ics.append((sig, float(ic)))

        if not signal_ics:
            preds.append({"horizon": f"{h}h", "predicted_return": None, "signal_strength": None,
                          "ci_low": None, "ci_high": None, "top_signals": []})
            continue

        top5 = sorted(signal_ics, key=lambda x: abs(x[1]), reverse=True)[:5]
        window = signals.tail(100)
        composite_score = 0.0
        weight_sum = 0.0
        
        for sig, ic in top5:
            col = window[sig].dropna()
            if len(col) < 10:
                continue
            z = (latest_signals.get(sig, np.nan) - col.mean()) / (col.std() + 1e-10)
            if np.isnan(z):
                continue
            composite_score += ic * z
            weight_sum += abs(ic)

        composite_score = composite_score / (weight_sum + 1e-10)
        recent_vol = fwd[h].dropna().std()
        predicted_return = float(composite_score * recent_vol) * 100 
        strength = min(100, max(0, int(50 + composite_score * 20)))
        ci_half = float(recent_vol) * 100
        
        preds.append({
            "horizon":          f"{h}h",
            "predicted_return": round(predicted_return, 3),
            "signal_strength":  strength,
            "ci_low":           round(predicted_return - ci_half, 3),
            "ci_high":          round(predicted_return + ci_half, 3),
            "top_signals":      [{"signal": s, "ic": round(ic, 3)} for s, ic in top5],
        })

    return preds


def _composite_score(predictions: list[dict], ic_rows: list[dict]) -> dict:
    strengths = [p["signal_strength"] for p in predictions if p["signal_strength"] is not None]
    score = int(sum(strengths) / len(strengths)) if strengths else 50

    ret_4h = next((p["predicted_return"] for p in predictions if p["horizon"] == "4h"), None)
    if ret_4h is None: bias = "neutral"
    elif ret_4h > 0.1: bias = "long"
    elif ret_4h < -0.1: bias = "short"
    else: bias = "neutral"

    patterns = [r["pattern"] for r in ic_rows[:10]]
    if patterns.count("always +") >= 4: regime = "Trending Bull"
    elif patterns.count("always -") >= 4: regime = "Trending Bear"
    elif patterns.count("grows") >= 3: regime = "Momentum"
    elif patterns.count("decays") >= 3: regime = "Mean Reversion"
    elif patterns.count("rev→mom") >= 2: regime = "Regime Shift"
    else: regime = "Mixed / Unclear"

    return {"score": score, "bias": bias, "regime": regime}


def _indicator_snapshot(df: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    KEY_INDICATORS = ["rsi_14", "macd_hist", "bb_position", "atr_norm", "obv_signal", "mfi"]
    latest = signals.iloc[-1]
    window = signals.tail(200)
    result = []

    for sig in KEY_INDICATORS:
        val = latest.get(sig, np.nan)
        if np.isnan(val):
            continue
        col = window[sig].dropna()
        pct = int((col < val).sum() / len(col) * 100) if len(col) > 0 else 50
        direction = "up" if pct > 60 else ("dn" if pct < 40 else "nu")
        result.append({
            "name":       sig,
            "value":      _safe_float(val),
            "percentile": pct,
            "direction":  direction,
        })

    return result


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    frontend = Path(__file__).parent / "alphadesq.html"
    if frontend.exists():
        return FileResponse(frontend, media_type="text/html")
    return {"error": "alphadesq.html not found"}


@app.get("/api/price/{ticker}")
async def get_price(ticker: str):
    ticker = ticker.upper()
    from datetime import datetime, timedelta
    for i in range(1, 5):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        try:
            res = client.get_daily_open_close_agg(ticker, target_date)
            if res and hasattr(res, "close"):
                return {"ticker": ticker, "date": target_date, "close": res.close}
        except Exception:
            continue
    raise HTTPException(status_code=404, detail=f"No recent price data for {ticker}")


@app.get("/api/dashboard/{ticker}")
async def get_dashboard(ticker: str):
    ticker = ticker.upper()
    t0 = time.perf_counter()

    try:
        df = _fetch_hourly(ticker, n_bars=600)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {e}")

    if len(df) < 100:
        raise HTTPException(status_code=422, detail=f"Insufficient data: only {len(df)} bars (need 100+)")

    signals = _compute_signals(df)

    ic_rows     = _compute_ic_rows(df, signals)
    predictions = _compute_predictions(df, signals)
    composite   = _composite_score(predictions, ic_rows)
    indicators  = _indicator_snapshot(df, signals)

    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "ticker":       ticker,
        "bar_count":    len(df),
        "computed_at":  pd.Timestamp.now().isoformat(),
        "elapsed_sec":  elapsed,
        "ic_table":     ic_rows,
        "predictions":  predictions,
        "composite":    composite,
        "indicators":   indicators,
    }

@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    ticker  = req.ticker.upper()
    horizon = req.horizon.lower().rstrip("h") if "h" in req.horizon.lower() else req.horizon
    h_int   = int(horizon) if horizon.isdigit() else 4

    dash = await get_dashboard(ticker)

    h_key   = f"ic_{h_int}h"
    top3    = sorted(
        [r for r in dash["ic_table"] if r.get(h_key) is not None],
        key=lambda r: abs(r[h_key]),
        reverse=True
    )[:3]

    pred = next((p for p in dash["predictions"] if p["horizon"] == f"{h_int}h"), None)
    composite = dash["composite"]

    bias_word = {"long": "BUY", "short": "SELL", "neutral": "HOLD"}[composite["bias"]]
    signal_lines = "\n".join(
        f"  • {r['signal']:18s}  IC={r[h_key]:+.3f}  pattern={r['pattern']}"
        for r in top3
    )
    pred_text = (
        f"Predicted {h_int}h return: {pred['predicted_return']:+.2f}%  "
        f"(CI [{pred['ci_low']:+.2f}%, {pred['ci_high']:+.2f}%])"
        if pred and pred["predicted_return"] is not None else "Prediction unavailable."
    )

    reasoning = (
        f"=== {ticker} / {h_int}h ANALYSIS ===\n"
        f"Decision: {bias_word}  |  Composite Score: {composite['score']}/100"
        f"  |  Regime: {composite['regime']}\n\n"
        f"Top 3 signals at {h_int}h:\n{signal_lines}\n\n"
        f"{pred_text}\n\n"
        f"Justification: The {composite['regime'].lower()} regime is supported by "
        f"{top3[0]['signal']} (IC {top3[0][h_key]:+.3f}, {top3[0]['pattern']}) "
        f"and {top3[1]['signal']} (IC {top3[1][h_key]:+.3f}, {top3[1]['pattern']}), "
        f"yielding a {bias_word.lower()} signal with strength {pred['signal_strength'] if pred else 'N/A'}/100."
    )

    return {**dash, "reasoning": reasoning, "decision": bias_word}

@app.get("/api/signal/{ticker}/{signal_name}")
async def get_signal_detail(ticker: str, signal_name: str):
    ticker = ticker.upper()
    try:
        df = _fetch_hourly(ticker, n_bars=600)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

    signals = _compute_signals(df)
    if signal_name not in signals.columns:
        raise HTTPException(status_code=404, detail=f"Unknown signal '{signal_name}'.")

    series = signals[signal_name].dropna()
    last10 = [_safe_float(v) for v in series.tail(10).tolist()]

    return {
        "ticker":  ticker,
        "signal":  signal_name,
        "latest":  _safe_float(series.iloc[-1]),
        "mean":    _safe_float(series.mean()),
        "std":     _safe_float(series.std()),
        "min":     _safe_float(series.min()),
        "max":     _safe_float(series.max()),
        "last_10": last10,
    }

@app.get("/api/signals")
async def list_signals():
    return {"groups": SIGNAL_GROUPS}

@app.post("/api/chat")
async def handle_chat(payload: ChatPayload):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing in .env")

    # Map standard chat roles to Gemini roles
    gemini_messages = []
    for m in payload.messages:
        role = "model" if m.role == "assistant" else "user"
        gemini_messages.append({
            "role": role,
            "parts": [{"text": m.content}]
        })

    data = {
        "systemInstruction": {
            "parts": [{"text": payload.system}]
        },
        "contents": gemini_messages,
        "generationConfig": {
            "maxOutputTokens": 1000,
        }
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=data)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        resp_data = resp.json()
        try:
            # Extract text from Gemini and format it so the frontend doesn't need to change
            reply_text = resp_data["candidates"][0]["content"]["parts"][0]["text"]
            return {"content": [{"type": "text", "text": reply_text}]}
        except (KeyError, IndexError):
            raise HTTPException(status_code=500, detail="Unexpected response format from Gemini")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=True)