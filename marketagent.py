import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel,set_tracing_disabled, AsyncOpenAI, function_tool, ModelSettings, SQLiteSession, set_tracing_export_api_key
from tavily import TavilyClient, AsyncTavilyClient
from datetime import datetime, timedelta, timezone
import yfinance as yf
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
tracing_api_key = os.environ.get("OPENAI_API_KEY")
set_tracing_export_api_key(tracing_api_key)

load_dotenv()
# set_tracing_disabled(True)
session = SQLiteSession("financial_chat_1", "conversation_history_1.db")


def get_secret(key: str):

    if key in st.secrets:
        return st.secrets[key]

    if os.getenv(key):
        return os.getenv(key)
    
    return None

# --- Load API keys ---
gemini_api_key = get_secret("GEMINI_API_KEY")
tavily_api_key = get_secret("TAVILY_API_KEY")
tracing_api_key = get_secret("OPENAI_API_KEY")

# Safety check
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in Streamlit secrets or environment.")

# gemini_api_key = os.environ.get("GEMINI_API_KEY")
# tavily_api_key = os.environ.get("TAVILY_API_KEY")


external_client: AsyncOpenAI = AsyncOpenAI(
   api_key=gemini_api_key,
   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

llm_model:OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
   model="gemini-2.5-flash",
   openai_client=external_client
)

def progress(msg: str):
   print(f"\n… {msg}", flush=True)

def _resolve_symbol(query: str) -> str:
    manual_map = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "tesla": "TSLA",
        "meta": "META",
        "amazon": "AMZN",
        "oracle": "ORCL",
        "netflix": "NFLX",
        "ibm": "IBM",
        "servicenow": "NOW",
        "s&p": "^GSPC",
        "nasdaq": "^IXIC",
        "dow jones": "^DJI",
        "byd": "1211.HK",
        "hyundai": "005380.KS",
        "toyota": "TM",
        "samsung": "005930.KQ",
        "sony": "SONY",
        "intel": "INTC",
        "qualcomm": "QCOM",
        "cisco": "CSCO",
        "paypal": "PYPL"
    }
    return manual_map.get(query.lower().strip(), query.upper())

@function_tool
async def resolve_symbol(query: str) -> str:
    return _resolve_symbol(query)

@function_tool
async def get_stock_data(symbol: str):
    """
    Fetch recent daily stock data using yfinance.
    """
    resolved = _resolve_symbol(symbol) 
    ticker = yf.Ticker(resolved)
    hist = ticker.history(period="1mo")
    if hist.empty:
        return {"error": f"No data found for {resolved}"}

    latest = hist.tail(1).iloc[0]
    return {
        "symbol": resolved,
        "latest_date": latest.name.strftime("%Y-%m-%d"),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"])
    }

@function_tool
async def get_top_companies(query: str = "Top companies in stock market today", limit: int = 5):
   """
   Use Tavily web search to fetch the latest list of top companies
   (by performance, market cap, etc.) depending on user request.
   """
   progress(f"Searching web for: {query}")
   tavily_client = AsyncTavilyClient(os.environ["TAVILY_API_KEY"])
   results = await tavily_client.search(query, max_results=limit)

   return {
       "query": query,
       "companies": [
           {"title": r["title"], "snippet": r["content"], "url": r["url"]}
           for r in results.get("results", [])
       ]
   }

@function_tool
async def fetch_market_stats(tickers: list[str]):
    """
    Fetch current market statistics (price, market cap, 52w high/low, PE, volume).
    Uses yfinance fast_info instead of .info (better for global tickers).
    """
    results = {}
    for ticker in tickers:
        t = yf.Ticker(ticker)
        fast = t.fast_info
        hist = t.history(period="5d")

        latest = hist.tail(1).iloc[0] if not hist.empty else None

        results[ticker] = {
            "currentPrice": float(fast.get("last_price", latest["Close"] if latest is not None else 0)),
            "marketCap": fast.get("market_cap"),
            "fiftyTwoWeekHigh": fast.get("year_high"),
            "fiftyTwoWeekLow": fast.get("year_low"),
            "forwardPE": fast.get("forward_pe"),
            "volume": int(latest["Volume"]) if latest is not None else fast.get("last_volume"),
            "previousClose": fast.get("previous_close"),
            "currency": fast.get("currency")
        }
    return results

@function_tool
async def get_stock_news(ticker: str):
   # NEW: print progress
   progress(f"Fetching news for {ticker} via Tavily…")
   tavily_client = AsyncTavilyClient(os.environ["TAVILY_API_KEY"])
   query = f"Latest stock news for {ticker} (past 14 days)"
   response = await tavily_client.search(query, max_results=3)
   progress(f"Done fetching news for {ticker}.")
   return response

@function_tool
async def get_market_data(symbol: str):
    """
    Fetch weekly stock data using yfinance.
    """
    resolved = await _resolve_symbol(symbol)
    ticker = yf.Ticker(resolved)
    hist = ticker.history(period="6mo", interval="1wk")
    if hist.empty:
        return {"error": f"No weekly data found for {resolved}"}

    latest = hist.tail(1).iloc[0]
    return {
        "symbol": resolved,
        "latest_week": latest.name.strftime("%Y-%m-%d"),
        "open": float(latest["Open"]),
        "high": float(latest["High"]),
        "low": float(latest["Low"]),
        "close": float(latest["Close"]),
        "volume": int(latest["Volume"])
    }

@function_tool
async def get_full_stock_report(company: str):
    """
    Resolve a company name → ticker → fetch both daily and weekly market data.
    Returns structured JSON with symbol, latest daily + weekly stats.
    """
    resolved = await _resolve_symbol(company)
    ticker = yf.Ticker(resolved)

    # --- Daily data (1 month) ---
    daily_hist = ticker.history(period="1mo")
    daily_data = None
    if not daily_hist.empty:
        latest = daily_hist.tail(1).iloc[0]
        daily_data = {
            "latest_date": latest.name.strftime("%Y-%m-%d"),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": int(latest["Volume"])
        }

    # --- Weekly data (6 months) ---
    weekly_hist = ticker.history(period="6mo", interval="1wk")
    weekly_data = None
    if not weekly_hist.empty:
        latest = weekly_hist.tail(1).iloc[0]
        weekly_data = {
            "latest_week": latest.name.strftime("%Y-%m-%d"),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
            "close": float(latest["Close"]),
            "volume": int(latest["Volume"])
        }

    return {
        "symbol": resolved,
        "company": company,
        "daily": daily_data,
        "weekly": weekly_data
    }

PlanningAgent = Agent(
   name="Planning Agent",
   instructions="""
       You are a planning agent.
       For each user request, decide:
       1. Which sub-agents (NewsAgent, MarketAgent) are needed.
       2. In what sequence to call them.
       3. What data each should fetch.
       Output a short JSON plan.
   """,
   model=llm_model,
   handoffs= ['FinancialAgent']
)

NewsAgent = Agent (
   name="News Agent",
   instructions="You are responsible for fetching the latest stock news and data when asked.",
   model=llm_model,
   tools=[get_stock_news]
)

MarketAgent = Agent(
   name= "MarketAgent",
   instructions="""
        You are responsible for fetching and analyzing stock/market data using Yahoo Finance (yfinance).
        Always think step by step:
        1. Clarify which market, index, or stock symbol the user cares about.
        2. Decide the most appropriate time period and interval.
        3. Use get_market_data or get_stock_data to fetch structured financial data.
        4. Use fetch_market_stats for current snapshot values (price, market cap, PE, etc.).
        5. Summarize key insights (current price, 52-week high/low, recent trend).
        Never provide news or speculation — just numbers and trends.
    """,
   model=llm_model,
   tools=[resolve_symbol, fetch_market_stats, get_market_data, get_stock_data, get_full_stock_report]
)

QAAssessmentAgent = Agent(
    name="QA Agent",
    instructions="""
        You are the Source Quality Assessment Agent.
        Your job is to evaluate the credibility of sources.
        - Prefer trusted outlets (.gov, .edu, Bloomberg, Reuters, WSJ, NYTimes, Investopedia).
        - Score each source from 0 to 1.
        - Add a short justification (e.g., 'Bloomberg is reliable financial outlet').
        Output JSON: [{url, score, justification}]
    """,
    model=llm_model,
)

ConflictDetectionAgent = Agent(
    name="Conflict Detection Agent",
    instructions="""
        You are the Conflict Detection Agent.
        Your job is to detect conflicting information between sources or agents.
        Compare numeric values (prices, metrics) and textual claims (buy/sell sentiment).
        - If conflicts exist, flag them clearly with both sides.
        - If consistent, return 'No major conflicts detected'.
        Output JSON: {conflicts: [...], notes: "..."}
    """,
    model=llm_model,
)

EducationAgent = Agent(
   name="Education Agent",
   instructions="""
       You are a financial educator.
       Your job is to explain investing concepts (mutual funds, ETFs, index funds, risk vs return, diversification, etc.)
       in simple, beginner-friendly language.
       Do not provide financial advice — just educational explanations.
   """,
   model=llm_model,
)

FinancialAgent = Agent(
   name="Financial Orchestrator",
   instructions=(
    """
        You are the base orchestrator.
        Your role is to decide which specialized agent to call for a given user query.
        Follow these rules:
        1. Understand the user request clearly. If unclear, ask one clarifying question.
        2. Compare the request against the available agents’ expertise and select the best fit.
        3. If multiple agents could handle it, choose the one with the most precise or specialized domain.
        4. If no agent is suitable, respond politely that no agent can handle the task.
        5. Always provide a short explanation of why you routed the request the way you did.
        6. If the called agent fails, suggest an alternative agent or handle gracefully.
        7. ALWAYS call resolve_symbol before calling NewsAgent or MarketAgent  unless the user already provided a ticker symbol (e.g., AAPL, MSFT).
        8. Use Educator agent to fetch answers to user queries related to finanical education.
        
        Special rule:
        If the user asks about "top companies", "top stocks", "top performers", directly use the get_top_companies tool (no need for specific symbols).

        After gathering results from NewsAgent/MarketAgent:
            → ALWAYS call the QA Agent to assess source quality.
            → Then call the Conflict Detection Agent to check for contradictions.
            → Merge their outputs before responding to the user.
    """
   ),
   model=llm_model,
   tools=[ resolve_symbol,
           PlanningAgent.as_tool(
               tool_name="planner",
               tool_description="You are a planning agent."
           ),
           NewsAgent.as_tool(
               tool_name='stock_market_news_reporter',
               tool_description='You are responsible for fetching the latest stock news and data when asked.'
            ),
           MarketAgent.as_tool(
               tool_name='market_data',
               tool_description='Fetch stock/market data using Alpha Vantage"'
            ),
            QAAssessmentAgent.as_tool(
                tool_name='qa_agent',
                tool_description='Evaluates source credibility and quality.'
            ),
            ConflictDetectionAgent.as_tool(
                tool_name='conflict_checker',
                tool_description='Detects conflicts or contradictions in gathered information.'
            ),
            EducationAgent.as_tool(
                tool_name="educator",
                tool_description="Explains general investment concepts in plain English"
            ),
            get_top_companies
        ]
)


async def main():
   print("=== Stock/Market Assistant ===")
   # Persistent session across all user turns
   global session
   while True:
        try:
            user_query = input("\nAsk me about stocks (or type 'exit' to quit): ")
            if user_query.lower() in ["exit", "quit"]:
               print("Goodbye 👋")
               break
           
                # with trace("Financial Agent Workflow"):
            result = await Runner.run(FinancialAgent, user_query, session=session)
          
            print("Agent response:\n", result.final_output)

        except KeyboardInterrupt:
            print("\nGoodbye 👋")
            break

if __name__ == "__main__":
   asyncio.run(main())



