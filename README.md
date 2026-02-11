Updates 2/11/2026
1. Now the tool can del with 6 different pairs of currency trading. Please    revise the variable "currency_pair" to see the effects. The 6 pairs are "EUR/USD", "USD/JPY", "GBP/USD", "USD/CNY", "USD/CAD", "AUD/USD".
2. Modified the news sources. Now in all 6 cases, investing.com can fetch 15 news and market watch can fetch 20 news. Yahoo Finance can fetch news only for EUR/USD.
3. Added Gemini model. You can fill in the gemini_api_key to get those news reported in Gemini as well. In default_config.json, you can decide on whether models to use in variable "llm_models". 

Updates 2/10/2026
1. Added the default_config.json to set up the initial values
	start_date and end_date in formats of yyyy-mm-dd; 
2. Added currency_pair
	currently there are three pairs available: EUR/USD; USD/JPN; GBP/USD
3. Added max_news
	set up the number of maximum news to collect in one run, default is None 

# Currency News Fetching & LLM Analysis

This small project crawls currencies related news directly from financial websites (no official news APIs), stores the results as JSON, and asks **DeepSeek** , **OpenAI** , **Gemini** to provide a trading-oriented analysis.

Current sources:
- Yahoo Finance (`https://finance.yahoo.com/currencies/eur-usd/`)
- Investing.com (`https://www.investing.com/currencies/eur-usd-news`) – best-effort scraping; selectors may need updates over time.

## 1. Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the same folder (you already have this) with:

```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here
```

Both keys are optional, but at least one of them must be valid if you want LLM analysis.

## 2. Usage

Basic run (last 7 days, both LLMs, default JSON file):

```bash
python news_fetching.py
```

Specify a date range (UTC, inclusive):

```bash
python news_fetching.py --start-date 2026-02-01 --end-date 2026-02-10
```

Choose which LLM provider to use:

```bash
# DeepSeek only
python news_fetching.py --provider deepseek

# OpenAI only
python news_fetching.py --provider openai

# Both (default)
python news_fetching.py --provider both
```

Limit how many news items are sent to the LLM (to keep prompts compact):

```bash
python news_fetching.py --max-articles 5
```

Change the JSON output path:

```bash
python news_fetching.py --output my_eur_usd_news.json
```

## 3. What the script does

1. **Crawls web pages** (no official news APIs) for EUR/USD related headlines and metadata.
2. **Filters by date range** if you provide `--start-date` / `--end-date` (otherwise uses last 7 days).
3. **Saves results to JSON** (titles, links, timestamps, and short content/summary).
4. For the first N items (see `--max-articles`), it **fetches the article pages** and extracts the **first 1–2 paragraphs**.
5. Sends that compact news context to **DeepSeek** and/or **OpenAI**, asking for:
   - Key market drivers
   - Short/medium-term EUR/USD outlook
   - Trading bias (bullish / bearish / sideways)
   - Main risks
6. Prints a **terminal report** with separate sections for each LLM provider.



