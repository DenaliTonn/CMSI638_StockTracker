import asyncio
import os
import sys
from dotenv import load_dotenv
import warnings

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

warnings.filterwarnings("ignore")
load_dotenv()

MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
PYTHON_BIN = sys.executable  # Safely uses the current uv environment

async def run_single_agent_task(agent: LlmAgent, user_input: str, session_service: InMemorySessionService) -> str:
    """Helper function to spin up an agent, send a single prompt, and return the raw text."""
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    session = await session_service.create_session(app_name=agent.name, user_id="user_local")
    content = types.Content(role="user", parts=[types.Part(text=user_input)])
    
    response_text = ""
    async for event in runner.run_async(user_id="user_local", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text
    return response_text

async def main():
    # 1. Initialize the MCP Connection
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=PYTHON_BIN, args=[MCP_SERVER_PATH])
        )
    )
    session_service = InMemorySessionService()

    # 2. Define the Sub-Agents
    data_agent = LlmAgent(
        name="data_agent",
        model="gemini-2.5-flash",
        instruction="You are a Data Technician. Your ONLY job is to call compute_all_indicators for the requested ticker and return the raw data block exactly as received.",
        tools=[toolset]
    )

    quant_agent = LlmAgent(
        name="quant_agent",
        model="gemini-2.5-flash",
        instruction="You are a Quant Analyst. Your ONLY job is to call compute_ic_table for the requested ticker and return the formatted table exactly as received.",
        tools=[toolset]
    )

    reasoning_agent = LlmAgent(
        name="reasoning_agent",
        model="gemini-2.5-flash", # Pro model recommended for synthesis/logic
        instruction="""You are a Quantitative Executioner. You will receive a Data Snapshot, an IC Table, and a Trading Horizon.
        
        Follow these strict rules:
        1. Find the top 3 signals in the IC Table with the highest absolute values for the requested horizon.
        2. Look up the current values of those specific 3 signals in the Data Snapshot.
        3. Issue a definitive decision: BUY, SELL, or HOLD. 
        4. Provide a concise 2-sentence justification explaining how those 3 signals align.
        """,
        tools=[] # No tools; it reasons over the context provided by the orchestrator
    )

    print("=== Multi-Agent Quant Orchestrator Ready ===")
    
    # 3. The Orchestrator Loop
    while True:
        ticker = input("\nWhich ticker would you like to analyze? (or 'quit'): ").strip().upper()
        if not ticker: continue
        if ticker in ("QUIT", "EXIT"): break

        # Step A: Data Retrieval
        print(f"\n[Orchestrator] 📊 Dispatching Data Agent for {ticker}...")
        data_result = await run_single_agent_task(data_agent, f"Get all indicators for {ticker}", session_service)

        # Step B: IC Computation
        print(f"[Orchestrator] 🧮 Dispatching Quant Agent for {ticker}...")
        quant_result = await run_single_agent_task(quant_agent, f"Get IC table for {ticker}", session_service)

        # Step C: Horizon Selection
        print("\n[Orchestrator] Data retrieved successfully.")
        horizon = ""
        while horizon not in ["1h", "4h", "8h", "24h"]:
            horizon = input("What trading window are you looking at? (1h, 4h, 8h, 24h): ").strip().lower()


        # Step D: Reasoning & Execution
        print(f"\n[Orchestrator] 🧠 Passing state to Reasoning Agent for {horizon} analysis...")
        await asyncio.sleep(5)
        reasoning_prompt = (
            f"TICKER: {ticker}\n"
            f"HORIZON: {horizon}\n\n"
            f"DATA SNAPSHOT:\n{data_result}\n\n"
            f"IC TABLE:\n{quant_result}"
        )
        
        decision = await run_single_agent_task(reasoning_agent, reasoning_prompt, session_service)
        
        print("\n================ FINAL DECISION ================")
        print(decision.strip())
        print("================================================\n")

if __name__ == "__main__":
    asyncio.run(main())