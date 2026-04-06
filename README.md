# CMSI638-stocktracker"

A Multi-agent stock analysis system built with Google ADK and FastMCP. The agent uses a local MCP server to expose stock analysis tools via Massive API and a Google Gemini-backed ADK agent to answer natural language queries about stocks and tickers.

---

## Requirements:
- Python 3.11+
- ['uv']
- A Massive API Key
- A Gemini API Key

---

## Setup:

### 1. Clone the repository
```bash
git clone <https://github.com/DenaliTonn/CMSI638_StockTracker.git>
cd CMSI638_StockTracker
```

### 2. Create and activate a virutal environment
 Using 'uv':
 ```bash
 uv venv
 source .venv/bin/activate
 ```

### 3. Install dependencies
```bash
uv pip install -e .
```
This install all dependencies declared in 'pyproject.toml'
| Package | Purpose |
|---|---|
| `fastmcp` | MCP server framework for exposing tools |
| `google-adk` | Google Agent Development Kit (agent runtime) |
| `google-genai` | Google Gemini model client |
| `mcp` | Model Context Protocol client/server primitives |
| `polygon` / `polygon-api-client` | Polygon.io stock market data |
| `python-dotenv` | Loads environment variables from `.env` |
| `massive` | NLU/intent classification support |

### 4. Configure environment variables
```bash
touch .env
```
Add your API keys:

```env
MASSIVE_API_KEY=your_massive_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

---

### Running the Agent
The project has two components that work together:
- **`main.py`** — the FastMCP server that exposes stock analysis tools
- **`adk_agent.py`** — the Google ADK agent that connects to the MCP server and handles user queries
 
Run the agent with:
 
```bash
uv run adk_agent.py
```