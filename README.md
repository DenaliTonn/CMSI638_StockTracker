# CMSI638 – StockTracker

A multi-agent stock analysis system built with **Google ADK** and **FastMCP**.  
The agent exposes **32 alpha signals** via a local MCP server, computes **Information Coefficient (IC) tables** across four time horizons (1h / 4h / 8h / 24h), and answers natural-language queries via a Gemini-backed ADK agent.

---

## Requirements

- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv)
- A **Massive API** key
- A **Gemini API** key

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/DenaliTonn/CMSI638_StockTracker.git
cd CMSI638_StockTracker
```

### 2. Create and activate a virtual environment
```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
uv pip install -e .
```

### 4. Configure environment variables
```bash
touch .env
```
```env
MASSIVE_API_KEY=your_massive_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
# MARKETAUX_API_KEY=your_key_here   # optional – for NLP sentiment (commented out)
```

### 5. Run the agent
```bash
uv run adk_agent.py
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `get_stock_price` | Most-recent daily closing price for a ticker |
| `calculate_momentum` | N-day Rate of Change (ROC) momentum factor |
| `compute_all_indicators` | Snapshot of all 32 alpha signals for the latest bar |
| `compute_ic_table` | IC-by-horizon table (1h/4h/8h/24h) with pattern classification |
| `get_signal_detail` | Last 10 values + stats for a single named signal |

---

## 32 Alpha Signals

| Group | Signals |
|---|---|
| Volume | `vol_7h`, `vol_35h`, `vol_147h`, `vol_accel`, `vol_ratio`, `vol_norm_ret` |
| Momentum | `momentum_7h`, `momentum_35h`, `momentum_147h`, `roc_7h`, `roc_35h` |
| RSI | `rsi_14`, `rsi_50` |
| MACD / PPO | `macd_hist`, `macd_signal`, `ppo` |
| Volatility | `atr_norm`, `hl_range` |
| MA Distance | `dist_ma35`, `dist_ma147`, `dist_ma441`, `pct_from_high` |
| Candlestick | `upper_wick`, `lower_wick`, `oc_direction`, `overnight_gap` |
| Other | `chaikin_ad`, `obv_signal`, `mfi`, `amihud`, `bb_position`, `sq_ret_lag1` |

---

## IC Table Output (example)

```
── GOOGL HOURLY – IC BY HORIZON ──
Signal              1h       4h       8h      24h  Best   Pattern
────────────────────────────────────────────────────────────────────────
vol_35h         +0.036   +0.089   +0.111   +0.121  24h    always +
vol_7h          +0.009   +0.054   +0.071   +0.141  24h    grows
chaikin_ad      -0.066   -0.096   -0.126   -0.141  24h    always -
momentum_147h   -0.029   -0.043   -0.048   -0.084  24h    always -
...
```

### Pattern key

| Pattern | Meaning | Best strategy fit |
|---|---|---|
| `always +` | Persistently positive IC at every horizon | Long bias at any holding period |
| `always -` | Persistently negative IC at every horizon | Short / hedge at any holding period |
| `grows` | IC strengthens over longer horizons | Swing / position trading |
| `decays` | IC strongest at short horizons, fades | Intraday / scalping |
| `rev→mom` | Negative IC short-term, positive long-term | Fade the move, then ride the trend |
| `mixed` | No clean pattern | Use with caution; regime-dependent |

---

## Dependency Overview

| Package | Purpose |
|---|---|
| `fastmcp` | MCP server framework |
| `google-adk` | Google Agent Development Kit |
| `google-genai` | Gemini model client |
| `mcp` | MCP client/server primitives |
| `massive` | Massive API market data client |
| `polygon` / `polygon-api-client` | Polygon.io market data |
| `numpy` | Numerical computation for signal math |
| `pandas` | Time-series data manipulation |
| `scipy` | Spearman rank correlation for IC calculation |
| `python-dotenv` | Environment variable loader |