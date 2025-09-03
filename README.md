# 📊 StoxFinlytics - Deep Search Market Agent

An AI-powered research assistant that performs deep search over financial markets, combining live data, web search, conflict detection, and financial education — all in one place.

## Set Up and Run
1. Clone the Repository
   ```bash
    uv init [folder_name] # name your folder in which the repository shall be cloned
    cd [folder_name]
    git clone https://github.com/AnxiousCodeGeek/deep-search-market-agent.git
    cd deep-search-market-agent
   ```
2. Create a Virtual Environment
   ```bash
    uv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows
   ```

3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```
4. Configure API Keys
  You need the following keys:
  - Google Gemini API (GEMINI_API_KEY)
  - Tavily Search API (TAVILY_API_KEY)
  - OpenAI API Key (optional, for tracing/debugging)

### Locally:

Create a ```.env``` file:
  ```
  GEMINI_API_KEY=your_key_here
  TAVILY_API_KEY=your_key_here
  OPENAI_API_KEY=your_key_here
  ```
## On Streamlit Cloud:

Put these in .streamlit/secrets.toml:
  ```toml
  GEMINI_API_KEY="your_key_here"
  TAVILY_API_KEY="your_key_here"
  OPENAI_API_KEY="your_key_here"
  ```
Run Locally
  ```bash
  streamlit run frontend.py
  ```
The app is deployed on: https://ai-market-agent.streamlit.app/

## Example Research Questions

Users can ask questions such as:

- 📈 “What’s the latest stock price and trend for NVIDIA?”

- 📰 “Give me the most recent financial news about Tesla.”

- 🏆 “Who are the top performing companies in the stock market today?”

- 🎓 “What is an ETF, and how does it differ from a mutual fund?”

- ⚖️ “Are there any conflicting reports about Tesla stock performance?”

## What Each Agent Does

**Planning Agent** → Decides which agents/tools to call for a user query.

**Financial Orchestrator** → The “main brain” that routes user queries to the right agents and merges results.

**News Agent** → Fetches the latest stock/financial news (via Tavily).

**Market Agent** → Fetches stock/market data from Yahoo Finance (yfinance).

**QA Assessment Agent** → Evaluates credibility of sources.

**Conflict Detection Agent** → Detects contradictory info between sources.

**Education Agent** → Explains finance/investment concepts in plain language.


## How the Team Coordinates

 - User interacts via a Streamlit chat interface.

 - Financial Orchestrator is the central coordinator:

   - Routes user queries to specialized agents.
  
   - Ensures data is consistent and trustworthy.
  
   - Calls the QA Agent and Conflict Detection Agent before final response.

 - Sub-agents (News, Market, Education) handle domain-specific tasks.

 - Results are merged, checked for conflicts, and presented back clearly to the user.

#### This multi-agent workflow ensures:

 - Accuracy (cross-verified data)

 - Transparency (quality scoring & conflict detection)

 - Clarity (simple explanations for complex financial concept

