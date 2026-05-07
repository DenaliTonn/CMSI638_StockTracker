"""
adk_agent.py — AlphaDeskQ Multi-Agent Orchestrator
====================================================
Uses Google ADK + Gemini to run a 3-agent pipeline:
  data_agent   → calls compute_all_indicators
  quant_agent  → calls compute_ic_table
  reasoning_agent → synthesizes BUY / SELL / HOLD + justification

Run standalone: uv run adk_agent.py
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
MCP_SERVER_PATH = str(Path(__file__).parent / "main.py")
PYTHON_BIN = sys.executable

HORIZONS = ("1h", "4h", "8h", "24h")
MODEL = "gemini-2.5-flash"

BANNER = """
╔══════════════════════════════════════════════════════╗
║          AlphaDeskQ  ·  Multi-Agent Quant            ║
║  data_agent → quant_agent → reasoning_agent         ║
╚══════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
#  Agent instructions
# ─────────────────────────────────────────────

DATA_AGENT_INSTRUCTION = """
You are a Data Technician. Your ONLY job is to call compute_all_indicators
for the requested ticker and return the raw data block EXACTLY as received —
no summarising, no commentary, no formatting changes.
"""

QUANT_AGENT_INSTRUCTION = """
You are a Quant Analyst. Your ONLY job is to call compute_ic_table
for the requested ticker and horizon and return the formatted table EXACTLY as received.
Do not add any commentary or modify the output.
"""

REASONING_AGENT_INSTRUCTION = """
You are a Quantitative Executioner. Be direct and concise — no prose padding.

SINGLE HORIZON — use this exact format, nothing more:

DECISION: <BUY|SELL|HOLD>  |  CONFIDENCE: <LOW|MEDIUM|HIGH>

SIGNALS @ <horizon>:
  1. <signal>  IC=<val>  Current=<val>  (<+/->)
  2. <signal>  IC=<val>  Current=<val>  (<+/->)
  3. <signal>  IC=<val>  Current=<val>  (<+/->)

THESIS: <One sentence explaining the convergence.>
RISK: <One sentence on the primary risk to monitor.>

MULTI-HORIZON — repeat the block above once per horizon, then add:

COMPARISON: <One sentence contrasting conviction across horizons.>
"""

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _parse_horizons(user_input: str) -> list[str]:
    """Extract all horizon mentions from a free-form user query."""
    found = re.findall(r'\b(1h|4h|8h|24h)\b', user_input.lower())
    seen = set()
    return [h for h in found if not (h in seen or seen.add(h))]


async def run_agent_task(
    agent: LlmAgent,
    prompt: str,
    session_service: InMemorySessionService,
) -> str:
    """Spin up a runner, send one prompt, collect the final text response."""
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    session = await session_service.create_session(
        app_name=agent.name, user_id="analyst"
    )
    content = types.Content(role="user", parts=[types.Part(text=prompt)])

    response_text = ""
    async for event in runner.run_async(
        user_id="analyst", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text

    return response_text.strip()


def _print_divider(label: str, text: str) -> None:
    width = 54
    print(f"\n{'─' * width}")
    print(f"  {label}")
    print(f"{'─' * width}")
    print(text)


def _get_input(prompt: str, valid: Optional[tuple] = None) -> str:
    while True:
        val = input(prompt).strip()
        if not valid or val.lower() in valid:
            return val.lower() if val else val
        print(f"  ⚠  Please enter one of: {', '.join(valid)}")

# ─────────────────────────────────────────────
#  Per-horizon data + quant pipeline
# ─────────────────────────────────────────────

async def orchestrate(
    ticker: str,
    horizon: str,
    toolset: McpToolset,
    session_service: InMemorySessionService,
) -> dict:
    """
    Run the data and quant agents for a single ticker + horizon.
    Returns dict with keys: ticker, horizon, data, ic_table.
    Reasoning agent is called separately in main() to support
    multi-horizon comparisons in a single consolidated prompt.
    """
    data_agent = LlmAgent(
        name="data_agent", model=MODEL,
        instruction=DATA_AGENT_INSTRUCTION, tools=[toolset]
    )
    quant_agent = LlmAgent(
        name="quant_agent", model=MODEL,
        instruction=QUANT_AGENT_INSTRUCTION, tools=[toolset]
    )

    print(f"\n  [1/2] 📊 Data Agent  — fetching signals for {ticker}...")
    data_result = await run_agent_task(
        data_agent, f"Get all indicators for {ticker}", session_service
    )

    await asyncio.sleep(3)  # prevent Massive API rate-limit on back-to-back fetches

    print(f"  [2/2] 🧮 Quant Agent — computing IC table for {ticker} at {horizon}...")
    ic_result = await run_agent_task(
        quant_agent, f"Compute the IC table for {ticker} at the {horizon} horizon", session_service
    )

    return {
        "ticker":   ticker,
        "horizon":  horizon,
        "data":     data_result,
        "ic_table": ic_result,
    }

# ─────────────────────────────────────────────
#  Main Orchestrator 
# ─────────────────────────────────────────────


async def main() -> None:
    print(BANNER)

    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=PYTHON_BIN, args=[MCP_SERVER_PATH])
        )
    )
    session_service = InMemorySessionService()

    while True:
        ticker = input("\nTicker (or 'quit'): ").strip().upper()
        if not ticker:
            continue
        if ticker in ("QUIT", "EXIT", "Q"):
            print("Goodbye.")
            break

        user_query = input("Query (e.g. 'analyze 8h' or 'compare 8h and 24h'): ").strip()
        if not user_query:
            continue

        horizons = _parse_horizons(user_query)

        # Fall back to explicit prompt if no horizons detected
        if not horizons:
            h = _get_input("No horizon detected. Enter one (1h / 4h / 8h / 24h): ", valid=HORIZONS)
            horizons = [h]

        try:
            # Run data + quant agents per horizon
            results = []
            for i, horizon in enumerate(horizons):
                print(f"\n── Horizon {i + 1}/{len(horizons)}: {horizon} ──")
                if i > 0:
                    await asyncio.sleep(3)  # rate-limit guard between horizon runs
                result = await orchestrate(ticker, horizon, toolset, session_service)
                results.append(result)

            # Build master prompt — single or comparative
            if len(results) == 1:
                r = results[0]
                reasoning_prompt = (
                    f"TICKER: {r['ticker']}\n"
                    f"HORIZON: {r['horizon']}\n\n"
                    f"DATA SNAPSHOT:\n{r['data']}\n\n"
                    f"IC TABLE:\n{r['ic_table']}"
                )
            else:
                blocks = []
                for r in results:
                    blocks.append(
                        f"── HORIZON: {r['horizon']} ──\n"
                        f"DATA SNAPSHOT:\n{r['data']}\n\n"
                        f"IC TABLE:\n{r['ic_table']}"
                    )
                reasoning_prompt = (
                    f"TICKER: {results[0]['ticker']}\n"
                    f"USER QUERY: {user_query}\n\n"
                    + "\n\n".join(blocks)
                )

            # Scale output tokens to number of horizons, floor at 1024
            output_tokens = max(400 * len(horizons) + 600, 1024)

            reasoning_agent = LlmAgent(
                name="reasoning_agent",
                model=MODEL,
                instruction=REASONING_AGENT_INSTRUCTION,
                tools=[],
                generate_content_config=types.GenerateContentConfig(
                    max_output_tokens=output_tokens,
                    temperature=0.1,
                )
            )

            print(f"\n[🧠] Reasoning Agent — synthesising {', '.join(horizons)} decision(s)...\n")
            await asyncio.sleep(3)  # rate-limit guard

            decision = await run_agent_task(reasoning_agent, reasoning_prompt, session_service)

        except Exception as exc:
            print(f"\n  ✗ Pipeline failed: {exc}")
            continue

        # Output
        for r in results:
            _print_divider(f"DATA SNAPSHOT  ·  {r['ticker']}  ·  {r['horizon']}", r["data"])
            _print_divider(f"IC TABLE  ·  {r['ticker']}  ·  {r['horizon']}", r["ic_table"])

        print(f"\n{'═' * 54}")
        print(f"  FINAL DECISION  ·  {ticker}  ·  {', '.join(horizons)}")
        print(f"{'═' * 54}")
        print(decision)
        print(f"{'═' * 54}\n")

        again = _get_input("Analyze another ticker? (y/n): ", valid=("y", "n"))
        if again == "n":
            print("Goodbye.")
            break


if __name__ == "__main__":
    asyncio.run(main())