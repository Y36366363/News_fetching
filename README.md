Updates 2/22/2026
1. Added a new benchmarking tool for the 3 API sources (NewsAPI / Alpha Vantage / EODHD):
   - `api_sources_benchmark.py`: benchmarks each FX pair by (a) newest-news lag, (b) coverage counts, and (c) a rule-based quality/reliability score (optional DeepSeek pass).
2. Benchmark runs now auto-log into:
   - `benchmarks/api_benchmark_log.jsonl` (JSONL append-only; can be disabled with `--no-log`).
3. Added a new config field to control whether to benchmark one pair or all 6 pairs:
   - `benchmark_fx_pairs`: set to `"EUR/USD"` (single) or `"all"` (all pairs).
4. Benchmark output now prints clearer time information:
   - run timestamp in both UTC and local time
   - per-provider “HQ newest” table and a separate “RAW newest” table (regardless of quality), with UTC + local timestamps.

Updates 2/21/2026
1. Added EODHD now saves two additional files (when enabled):
   - `{pair}_eodhd_content.json`: append-only content store (de-duplicated by URL; adds only newly seen articles)
   - `{pair}_eodhd_quality.json`: DeepSeek relevance/quality classification for the current run (counts printed once to terminal). The final LLM report filters EODHD items to related URLs.
   - with the alpha vantage accessable, please add your EODHD api in the .env file for the settings.

Updates 2/20/2026
1. Added contrains for a limit of maximum news to collect for a run for a single website source (investing, marketwatch, yahoo), it also can be set up the value of "null" with no limits. 
2. Added Alpha Vantage now saves two additional files (when enabled):
   - `{pair}_alphavantage_content.json`: append-only content store (de-duplicated by URL; adds only newly seen articles)
   - `{pair}_alphavantage_quality.json`: DeepSeek relevance/quality classification for the current run (counts printed once to terminal). The final LLM report filters Alpha Vantage items to related URLs.
   - with the alpha vantage accessable, please add your alpha vantage api in the .env file for the settings.

Updates 2/19/2026
1. Added a DeepSeek pre-analysis step for fx pairs to (a) compare news across sources, (b) rate news quality (relevance/credibility/signal/timeliness), (c) detect duplicates, and (d) recommend the best subset of articles before generating the final analysis report. The results are saved under `pre_analysis` in the output JSON.
2. Supported FX pairs are: `"EUR/USD"`, `"USD/JPY"`, `"GBP/USD"`, `"USD/CNY"`, `"USD/CAD"`, `"AUD/USD"`. Change `currency_pair` in `default_config.json` to analyze a different pair.
3. NewsAPI now saves two additional FX files:
   - `{pair}_newsapi_content.json`: append-only content store (de-duplicated by URL; adds only newly seen articles)
   - `{pair}_newsapi_quality.json`: DeepSeek relevance classification for the current run (counts printed once to terminal). The final LLM report uses only the related NewsAPI articles.
4. Each enabled LLM provider also saves its terminal output to a separate JSON file:
   - `{pair}_deepseek.json`, `{pair}_chatgpt.json`, `{pair}_gemini.json` once the targed LLM model is selected to output a report.

Updates 2/18/2026
1. Now with each run, the news collected from news api can be seen in another json file. For example, besides the basic information storaged in the file eur_usd_news.json, there is another file generated "eur_usd_news.json". In the column of "content", you can see the exact contents of the news collected. Since there is no valid access of contents to the cratching websites, only fx pairs of news api are available now.

Updates 2/17/2026
1. Now the output analysis for each LLM model can also be seen in the corresponding file. For example, for stock "AMZN", at the very end of the file, amzn_news.json, there is a new configuration "analysis" that storages the analysis generating from the model we use. If any of these models are forbidden in default_config.json, they will not be shown with each run.

Updates 2/16/2026
1. Added a new news-fetching source: news api, by using the api key of news api (100 calls per day). Now only 6 fx pairs can use this source.
2. Added parameters in default_config.json for the usage of the parameter of each news source. Now you can determine which source to use or not.

Updates 2/12/2026
1. Tested the Gemini to make sure that it works for analysis generation.
2. Added nasdaq 100 stocks to analyze, to do this, change "currency" to "stock" in default_config.json and set up the label name of the stock in "stock".
3. Tested Yahoo Finance for stocks and can gather about 78 news for a single stock.

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
	set up the number of maximum news to collect in one run, default is null 

# News Fetching & LLM Trading Analysis (FX + NASDAQ stocks)

This project crawls trading-related news directly from public web pages (no official news APIs required), saves results to JSON, and optionally asks **DeepSeek**, **OpenAI**, and/or **Gemini** to generate a trader-oriented analysis.

## Sources

- **Investing.com**: currency pages work well; some equity pages require browser-like headers (Cloudflare protection).
- **MarketWatch**: quote-related news.
- **Yahoo Finance**:
  - **Stocks**: uses the **RSS feed** (more stable than JS-rendered pages).
  - **FX**: best-effort HTML scraping.
- **NewsAPI.org** (FX only): API-based news collection with optional full-content store.
- **Alpha Vantage**: API-based financial news via `NEWS_SENTIMENT` (stocks + FX).
- **EODHD**: API-based financial news via EODHD News API (FX + stocks).

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file (you already have one) with any keys you want to use:

```bash
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
NEWS_API_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
```

You can run with **one** provider enabled, or multiple.

## Configuration (`default_config.json`)

This project is config-driven. The most important fields are:

- **`asset_type`**: `"currency"` or `"stock"`
- **`currency_pair`**: e.g. `"EUR/USD"` (used when `asset_type="currency"`)
- **`benchmark_fx_pairs`**: `"EUR/USD"` (single) or `"all"` (benchmark all 6 FX pairs) for `api_sources_benchmark.py`
- **`stock`**: e.g. `"AAPL"` (used when `asset_type="stock"`)
- **`start_date` / `end_date`**: `"YYYY-MM-DD"` (UTC, inclusive)
- **`max_news`**: maximum number of news items to collect per run (`null` = no limit)
- **`investing_max_items`**: max items from Investing crawler (`null` = no limit; still respects `max_news`)
- **`marketwatch_max_items`**: max items from MarketWatch crawler (`null` = no limit; still respects `max_news`)
- **`yahoo_max_items`**: max items from Yahoo crawler (`null` = no limit; still respects `max_news`)
- **`newsapi_max_items`**: max items from NewsAPI (`null` = no limit; still respects `max_news`)
- **`enable_pre_analysis`**: if `true`, uses DeepSeek to compare sources + rate article quality before the main report
- **`pre_analysis_max_articles`**: how many collected articles are sent to DeepSeek for the pre-analysis step
- **`pre_analysis_filter_for_report`**: if `true`, the main report LLM(s) will only see the `recommended_news_ids` subset from pre-analysis
- **`llm_models`**: enable/disable each model with `{"enabled": true/false}`

Example (FX mode):

```json
{
  "start_date": "2026-02-05",
  "end_date": "2026-02-12",
  "asset_type": "currency",
  "currency_pair": "EUR/USD",
  "stock": "AAPL",
  "max_news": null,
  "investing_max_items": null,
  "marketwatch_max_items": null,
  "yahoo_max_items": null,
  "newsapi_max_items": 10,
  "llm_models": {
    "deepseek": {"enabled": true},
    "openai": {"enabled": false},
    "gemini": {"enabled": false}
  }
}
```

Example (stock mode):

```json
{
  "start_date": "2026-02-05",
  "end_date": "2026-02-12",
  "asset_type": "stock",
  "currency_pair": "EUR/USD",
  "stock": "AAPL",
  "max_news": null,
  "investing_max_items": null,
  "marketwatch_max_items": null,
  "yahoo_max_items": null,
  "llm_models": {
    "deepseek": {"enabled": true},
    "openai": {"enabled": false},
    "gemini": {"enabled": true}
  }
}
```

## Usage

Basic run (reads `default_config.json`):

```bash
python news_fetching.py
```

Optional overrides:

- **Date range**:

```bash
python news_fetching.py --start-date 2026-02-01 --end-date 2026-02-10
```

- **Limit how many news items are sent to the LLM** (prompt size control):

```bash
python news_fetching.py --max-articles 5
```

- **Only crawl + save JSON (skip LLM analysis)**:

```bash
python news_fetching.py --no-llm
```

- **Verbose mode** (helps debugging blocked pages / HTTP issues):

```bash
python news_fetching.py --verbose
```

## API source benchmarking (`api_sources_benchmark.py`)

This tool benchmarks the 3 API providers (**NewsAPI**, **Alpha Vantage**, **EODHD**) for FX pairs.
It reports:
- **Recency**: newest item lag, plus newest *high-quality* item lag (rule-based filter)
- **Coverage**: item counts and unique host counts
- **Time clarity**: prints both **UTC** and **local** timestamps (lag is computed at run time)

### Benchmark (no API calls, uses cached content stores)

If you already have content stores from `news_fetching.py` (e.g. `eur_usd_newsapi_content.json`),
you can benchmark without consuming API quota:

```bash
python api_sources_benchmark.py --from-cache --cache-root . --window-days 30
```

### Benchmark (live API calls)

```bash
python api_sources_benchmark.py --window-days 7
```

### Logging

Each run appends to:
- `benchmarks/api_benchmark_log.jsonl`

Disable logging:

```bash
python api_sources_benchmark.py --no-log
```

### FX pair selection

- Default: uses `benchmark_fx_pairs` in `default_config.json` (`"EUR/USD"` or `"all"`).
- Override from CLI:

```bash
python api_sources_benchmark.py --pairs "EUR/USD,GBP/USD"
```

## Output

- Saves crawled items to a JSON file named after the selected asset:
  - FX: `eur_usd_news.json`, `usd_cny_news.json`, ...
  - Stock: `aapl_news.json`, ...
- Each item includes: `title`, `url`, `published_at` (if detected), `snippet`, `source`, `asset_type`

## URL lists (`asset_urls.py`)

All supported asset URLs live in `asset_urls.py`:

- **`SUPPORTED_PAIRS`**: currency pair source URLs
- **`SUPPORTED_STOCKS`**: NASDAQ symbols you provided
  - MarketWatch + Yahoo URLs are generated automatically by symbol
  - Investing equity **news** URLs are slug-based, so only some symbols have an `investing_url` override

If you want more Investing.com stock coverage, add more entries to the `_INVESTING_NEWS_OVERRIDES` map inside `asset_urls.py`.
