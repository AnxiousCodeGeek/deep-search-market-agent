import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel,set_tracing_disabled, AsyncOpenAI, function_tool, ModelSettings, SQLiteSession, set_tracing_export_api_key
from tavily import TavilyClient, AsyncTavilyClient
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta, timezone
import warnings
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
tracing_api_key = os.environ.get("OPENAI_API_KEY")
set_tracing_export_api_key(tracing_api_key)

load_dotenv()
# set_tracing_disabled(True)
session = SQLiteSession("financial_chat", "conversation_history.db")

gemini_api_key = os.environ.get("GEMINI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")
alpha_api_key = os.environ.get("ALPHA_API_KEY")

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

@function_tool
async def resolve_symbol(query: str) -> str:
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
       "byd" : "1211.HK"
   }
   return manual_map.get(query.lower().strip(), query.upper())

@function_tool
async def get_stock_data(symbol: str):
    """
    Fetch recent daily stock prices using Alpha Vantage API.
    """
    resolved = await resolve_symbol(symbol)
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={resolved}&apikey={alpha_api_key}&outputsize=compact"
    r = requests.get(url)
    data = r.json()

    if "Time Series (Daily)" not in data:
        return {"error": f"Alpha Vantage failed for {resolved}", "raw": data}

    ts = data["Time Series (Daily)"]
    latest_date = sorted(ts.keys())[-1]
    latest = ts[latest_date]

    return {
        "symbol": resolved,
        "latest_date": latest_date,
        "open": float(latest["1. open"]),
        "high": float(latest["2. high"]),
        "low": float(latest["3. low"]),
        "close": float(latest["4. close"]),
        "volume": int(latest["5. volume"])
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
    Fetch weekly time series for broader trend analysis.
    """
    resolved = await resolve_symbol(symbol)
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={resolved}&apikey={alpha_api_key}"
    r = requests.get(url)
    data = r.json()

    if "Weekly Time Series" not in data:
        return {"error": f"Alpha Vantage weekly data failed for {resolved}", "raw": data}

    ts = data["Weekly Time Series"]
    latest_date = sorted(ts.keys())[-1]
    latest = ts[latest_date]

    return {
        "symbol": resolved,
        "latest_week": latest_date,
        "open": float(latest["1. open"]),
        "high": float(latest["2. high"]),
        "low": float(latest["3. low"]),
        "close": float(latest["4. close"]),
        "volume": int(latest["5. volume"])
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
   tools=[resolve_symbol, get_stock_news]
)

MarketAgent = Agent(
   name= "MarketAgent",
   instructions=
       """
       You are responsible for fetching and analyzing stock/market data using Alpha Vantage.
       Always think step by step:
       1. Clarify which market, index, or stock symbol the user cares about.
       2. If the user provides a company name (e.g., 'Apple', 'Nvidia'),
          always call the resolve_symbol tool first to get the correct ticker (e.g., 'AAPL', 'NVDA').
       3. Decide the most appropriate time period and interval.
       4. Use the get_market_data tool to fetch structured financial data.
       5. Summarize key insights (e.g., current price, recent trend).
       Never provide news or speculation — just numbers and trends.
       """,
   model=llm_model,
   tools=[resolve_symbol, get_market_data, get_stock_data]
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
        7. Always call resolve_symbol before calling get_market_data or get_stock_news, unless the user already provided a ticker symbol (e.g., AAPL, MSFT).


        Special rule:
        If the user asks about "top companies", "top stocks", "top performers",
        directly use the get_top_companies tool (no need for specific symbols).


        After gathering results from NewsAgent/MarketAgent:
        → ALWAYS call the QA Agent to assess source quality.
        → Then call the Conflict Detection Agent to check for contradictions.
        → Merge their outputs before responding to the user.
    """
   ),
   model=llm_model,
   tools=[
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
       get_top_companies, resolve_symbol
   ]
)

# user_chat: list[dict] = []

async def main():
   print("=== Stock/Market Assistant ===")

   # Persistent session across all user turns
   global session
   # global user_chat

   while True:
        try:
            user_query = input("\nAsk me about stocks (or type 'exit' to quit): ")
            if user_query.lower() in ["exit", "quit"]:
               print("Goodbye 👋")
               break
           
            # with trace("Financial Agent Workflow"):
            result = await Runner.run(FinancialAgent,user_query, session=session)
          
           # query_history = {'role':'user','content':user_query}
           # user_chat.append(query_history)
           # result = await Runner.run(FinancialAgent, user_chat)
           # user_chat = result.to_input_list()
        #    print("\n=== REPORT ===\n")
        #    print("Agent called:", result.last_agent.name)
            print("Agent response:\n", result.final_output)

        except KeyboardInterrupt:
            print("\nGoodbye 👋")
            break

if __name__ == "__main__":
   asyncio.run(main())



