from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types
import asyncio
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
MCP_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
PYTHON_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin", "python")

async def main():
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=PYTHON_BIN,
                args=[MCP_SERVER_PATH]
            )
        )
    )
    
    agent = LlmAgent(
        name="finance_agent",
        model="gemini-2.5-flash",
        instruction="You are a financial assistant. Use your tools to answer questions.",
        tools=[toolset]
    )
    
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="finance_agent", session_service=session_service)
    session = await session_service.create_session(app_name="finance_agent", user_id="user1")
    
    print("Finance Agent is ready! (Type 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter a company or ticker: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        content = types.Content(role="user", parts=[types.Part(text=user_input)])
        
        async for event in runner.run_async(
            user_id="user1", session_id=session.id, new_message=content
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent: {part.text.strip()}")

if __name__ == "__main__":
    asyncio.run(main())