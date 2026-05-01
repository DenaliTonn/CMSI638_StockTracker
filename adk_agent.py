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

# ── Config ──────────────────────────────────────────────────────────────────
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

# ── Agent definitions ────────────────────────────────────────────────────────

DATA_AGENT_INSTRUCTION = """
You are a Data Technician. Your ONLY job is to call compute_all_indicators
for the requested ticker and return the raw data block EXACTLY as received —
no summarising, no commentary, no formatting changes.
"""

QUANT_AGENT_INSTRUCTION = """
You are a Quant Analyst. Your ONLY job is to call compute_ic_table
for the requested ticker and return the formatted table EXACTLY as received.
Do not add any commentary or modify the output.
"""

REASONING_AGENT_INSTRUCTION = """
You are a Quantitative Executioner on a sell-side equities desk.
You receive: TICKER, HORIZON, a Data Snapshot, and an IC Table.

STRICT RULES:
1. Identify the top 3 signals in the IC Table with the HIGHEST ABSOLUTE IC
   values at the requested HORIZON.
2. Look up the current values of those exact 3 signals in the Data Snapshot.
3. Issue one of: BUY / SELL / HOLD
   - BUY  if the weighted signal composite is positive and material (>0.03)
   - SELL if the weighted signal composite is negative and material (<-0.03)
   - HOLD otherwise
4. Output in this EXACT format (no deviation):

DECISION: <BUY|SELL|HOLD>
CONFIDENCE: <LOW|MEDIUM|HIGH>

TOP SIGNALS @ <horizon>:
  1. <signal_name>  IC=<value>  Current=<value>  Direction=<+/->
  2. <signal_name>  IC=<value>  Current=<value>  Direction=<+/->
  3. <signal_name>  IC=<value>  Current=<value>  Direction=<+/->

JUSTIFICATION:
<Two precise sentences explaining how the 3 signals converge to support the decision.>

RISK NOTE:
<One sentence on the primary risk or conflicting signal to monitor.>
"""

# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _prompt(label: str, text: str) -> None:
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


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def orchestrate(ticker: str, horizon: str, toolset: McpToolset,
                      session_service: InMemorySessionService) -> dict:
    """
    Run the full 3-agent pipeline for a ticker + horizon.
    Returns a dict with keys: ticker, horizon, data, ic_table, decision.
    """
    # Agent construction is cheap — rebuild per run to avoid stale session state
    data_agent = LlmAgent(
        name="data_agent", model=MODEL,
        instruction=DATA_AGENT_INSTRUCTION, tools=[toolset]
    )
    quant_agent = LlmAgent(
        name="quant_agent", model=MODEL,
        instruction=QUANT_AGENT_INSTRUCTION, tools=[toolset]
    )
    reasoning_agent = LlmAgent(
        name="reasoning_agent", model=MODEL,
        instruction=REASONING_AGENT_INSTRUCTION, tools=[]
    )

    print(f"\n[1/3] 📊 Data Agent  — fetching 32 signals for {ticker}...")
    data_result = await run_agent_task(
        data_agent, f"Get all indicators for {ticker}", session_service
    )

    print(f"[2/3] 🧮 Quant Agent — computing IC table for {ticker}...")
    ic_result = await run_agent_task(
        quant_agent, f"Compute the IC table for {ticker}", session_service
    )

    print(f"[3/3] 🧠 Reasoning Agent — synthesising {horizon} decision...\n")
    await asyncio.sleep(3)   # brief pause to avoid Gemini rate-limit on rapid sequential calls

    reasoning_prompt = (
        f"TICKER: {ticker}\n"
        f"HORIZON: {horizon}\n\n"
        f"DATA SNAPSHOT:\n{data_result}\n\n"
        f"IC TABLE:\n{ic_result}"
    )
    decision = await run_agent_task(
        reasoning_agent, reasoning_prompt, session_service
    )

    return {
        "ticker":   ticker,
        "horizon":  horizon,
        "data":     data_result,
        "ic_table": ic_result,
        "decision": decision,
    }


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

        horizon = _get_input(
            "Horizon (1h / 4h / 8h / 24h): ",
            valid=HORIZONS
        )
        if not horizon:
            continue

        try:
            result = await orchestrate(ticker, horizon, toolset, session_service)
        except Exception as exc:
            print(f"\n  ✗ Pipeline failed: {exc}")
            continue

        _prompt(f"DATA SNAPSHOT  ·  {ticker}", result["data"])
        _prompt(f"IC TABLE  ·  {ticker}", result["ic_table"])

        print(f"\n{'═' * 54}")
        print(f"  FINAL DECISION  ·  {ticker}  ·  {horizon}")
        print(f"{'═' * 54}")
        print(result["decision"])
        print(f"{'═' * 54}\n")

        again = _get_input("Analyze another ticker? (y/n): ", valid=("y", "n"))
        if again == "n":
            print("Goodbye.")
            break


if __name__ == "__main__":
    asyncio.run(main())