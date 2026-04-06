import os
from datetime import datetime, timedelta
from fastmcp import FastMCP
from massive import RESTClient
from dotenv import load_dotenv

load_dotenv()
mcp = FastMCP("StockAnalyzer")
client = RESTClient(os.getenv("MASSIVE_API_KEY"))

@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Fetches the most recent closing price for a ticker."""
    try:
        ticker = ticker.upper()
        for i in range(1, 5):
            target_date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            try:
                res = client.get_daily_open_close_agg(ticker, target_date)
                if res and hasattr(res, 'close'):
                    return f"The closing price for {ticker} on {target_date} was ${res.close:.2f}."
            except Exception:
                continue
        return f"No data found for {ticker} in the last 7 days."
    except Exception as e:
        return f"Error retrieving {ticker}: {str(e)}"

if __name__ == "__main__":
    mcp.run()