"""
End-to-end EUR/USD news collection and LLM analysis script.

Goals:
- Crawl finance news related to EUR/USD directly from web pages
  (no official news APIs).
- Optionally filter by a date range.
- Store the results as a JSON document (titles, URLs, publish times, snippets).
- Feed a compact subset of the news to DeepSeek and/or OpenAI so they
  can analyze the situation and provide trading suggestions.
- Print the analysis in a visually clear, readable format.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dotenv import load_dotenv
from openai import OpenAI


# Configuration

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}

# Supported currency pairs and their news URLs.
SUPPORTED_PAIRS: Dict[str, Dict[str, str]] = {
    "EUR/USD": {
        "investing_url": "https://www.investing.com/currencies/eur-usd-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/eurusd",
        "yahoo_url": "https://finance.yahoo.com/quote/EURUSD=X/",
    },
    "USD/JPY": {
        "investing_url": "https://www.investing.com/currencies/usd-jpy-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/usdjpy",
        "yahoo_url": "https://finance.yahoo.com/quote/JPY=X/",
    },
    "GBP/USD": {
        "investing_url": "https://www.investing.com/currencies/gbp-usd-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/gbpusd",
        "yahoo_url": "https://finance.yahoo.com/quote/GBPUSD=X/",
    },
    "USD/CNY": {
        "investing_url": "https://www.investing.com/currencies/usd-cny-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/usdcny",
        "yahoo_url": "https://finance.yahoo.com/quote/CNY=X/",
    },
    "USD/CAD": {
        "investing_url": "https://www.investing.com/currencies/usd-cad-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/usdcad",
        "yahoo_url": "https://finance.yahoo.com/quote/CAD=X/",
    },
    "AUD/USD": {
        "investing_url": "https://www.investing.com/currencies/aud-usd-news",
        "marketwatch_url": "https://www.marketwatch.com/investing/currency/audusd",
        "yahoo_url": "https://finance.yahoo.com/quote/AUDUSD=X/",
    },
}

# Path to default configuration (date range etc.), next to this script.
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

# Global flag to control verbose output (warnings, low-level errors).
VERBOSE = False


@dataclass
class NewsItem:
    id: int
    currency_pair: str
    source: str
    title: str
    url: str
    published_at: Optional[str]  # ISO 8601 string or None
    snippet: str                 # short content or first 1–2 paragraphs


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def debug_log(message: str) -> None:
    """
    Print low-importance warnings / errors only when VERBOSE is enabled.
    """
    if VERBOSE:
        print(message)


def parse_cli_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"Invalid date format: {value!r}. Use YYYY-MM-DD.")


def parse_iso_like_datetime(value: str) -> Optional[datetime]:
    """
    Parse ISO-like datetime strings (e.g. Yahoo Finance time tags) into UTC.
    Falls back to python-dateutil for robustness.
    """
    try:
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    except Exception:
        try:
            dt = date_parser.parse(value)
        except Exception:
            return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def fetch_html(url: str, retries: int = 3, delay: int = 2) -> Optional[str]:
    """
    Fetch HTML content with simple retry and backoff.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return resp.text
        except requests.RequestException as exc:
            debug_log(
                f"[WARN] Failed to fetch {url!r} "
                f"(attempt {attempt}/{retries}): {exc}"
            )
            time.sleep(delay * attempt)
    debug_log(f"[ERROR] Giving up on {url!r} after {retries} attempts.")
    return None


def fetch_article_snippet(url: str, max_paragraphs: int = 2) -> str:
    """
    Fetch a news article page and return the first 1–2 meaningful paragraphs.
    This keeps prompts short while still informative for the LLM.
    """
    html = fetch_html(url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    article = soup.find("article")
    root = article if article is not None else (soup.body or soup)

    paragraphs: List[str] = []
    for p in root.find_all("p"):
        txt = p.get_text(strip=True)
        if not txt or len(txt) < 40:
            # Skip very short / boilerplate fragments
            continue
        paragraphs.append(txt)
        if len(paragraphs) >= max_paragraphs:
            break

    return "\n\n".join(paragraphs[:max_paragraphs])


def within_date_range(dt: Optional[datetime],
                      start_date: Optional[date],
                      end_date: Optional[date]) -> bool:
    """
    Check if a datetime lies between start_date and end_date (inclusive),
    comparing only the calendar date portion.
    """
    if dt is None:
        # If we cannot parse a date, still keep the item. Many sites show
        # relative times like "2 hours ago" or omit explicit dates. To avoid
        # silently dropping everything, we do not enforce date filters when
        # the article time is unavailable.
        return True

    d = dt.date()
    if start_date and d < start_date:
        return False
    if end_date and d > end_date:
        return False
    return True


# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------

def scrape_investing_pair(
    pair: str,
    url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a currency pair's news from Investing.com.
    This may return 0 items if the site layout or anti-bot protection changes.
    """
    print(f"\n[INFO] Crawling Investing.com ({pair} news)...")
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    # The HTML layout on Investing.com changes over time. Instead of relying on
    # specific classes, we use a generic strategy:
    # - Find all links that point to /news/ (economy, forex, etc.)
    # - Treat their text as the article title.
    # We then optionally try to find a nearby date string, but missing dates will
    # NOT cause the item to be dropped (see within_date_range).
    for a in soup.select("a[href*='/news/']"):
        try:
            title = a.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.investing.com" + href

            # Try to find a small date string near the link (e.g. "2 hours ago").
            # If we cannot parse it, we still keep the item.
            published_dt: Optional[datetime] = None
            parent = a.parent
            if parent is not None:
                # Look a bit around the anchor for time / date information
                time_tag = parent.find("time")
                if time_tag and time_tag.get_text(strip=True):
                    published_dt = parse_iso_like_datetime(time_tag.get_text(strip=True))

            if not within_date_range(published_dt, start_date, end_date):
                continue

            # We don't rely on the on-page snippet; we will later fetch the
            # article and extract the first 1–2 paragraphs. Store a very short
            # inline snippet as a fallback.
            snippet = ""
            sibling_p = a.find_next("p")
            if sibling_p:
                snippet = sibling_p.get_text(strip=True)[:600]

            items.append(
                {
                    "source": "Investing.com",
                    "currency_pair": pair,
                    "title": title,
                    "url": href,
                    "published_dt": published_dt,
                    "snippet": snippet,
                }
            )

            if max_items is not None and len(items) >= max_items:
                break
        except Exception:
            continue

    print(f"[INFO] Investing.com items collected: {len(items)}")
    return items


def scrape_marketwatch_pair(
    pair: str,
    url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a currency pair's news section on MarketWatch.
    We target links with the "mod=mw_quote_news" marker, which are specifically
    attached to quote-related news items.
    """
    print(f"\n[INFO] Crawling MarketWatch ({pair} news)...")
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    # Links for the quote's news section typically contain this marker.
    for a in soup.select("a[href*='mod=mw_quote_news']"):
        try:
            title = a.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.marketwatch.com" + href

            # MarketWatch often shows the datetime as nearby text, but it may not
            # be in a dedicated <time> tag. For simplicity and robustness, we
            # leave publish time as None and let within_date_range() keep it.
            published_dt: Optional[datetime] = None

            if not within_date_range(published_dt, start_date, end_date):
                continue

            snippet = ""
            sibling_p = a.find_next("p")
            if sibling_p:
                snippet = sibling_p.get_text(strip=True)[:600]

            items.append(
                {
                    "source": "MarketWatch",
                    "currency_pair": pair,
                    "title": title,
                    "url": href,
                    "published_dt": published_dt,
                    "snippet": snippet,
                }
            )

            if max_items is not None and len(items) >= max_items:
                break
        except Exception:
            continue

    print(f"[INFO] MarketWatch items collected: {len(items)}")
    return items


def scrape_yahoo_pair(
    pair: str,
    url: str,
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a currency pair's quote page on Yahoo Finance.

    We do NOT rely on a stable DOM or exact "news" section selector, since
    Yahoo frequently changes layouts and uses heavy client-side rendering.
    Instead, we:
      - Look for <a> tags that link to /news/ articles.
      - Treat their text as the title.
    Date information on the quote page is often shown as relative labels like
    "2d ago"; to keep this scraper robust, we currently leave publish time as
    None and let the LLM focus on article content.
    """
    print(f"\n[INFO] Crawling Yahoo Finance ({pair} quote news)...")
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    for a in soup.select("a[href*='/news/']"):
        try:
            title = a.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://finance.yahoo.com" + href

            # Publish time is left as None for robustness.
            published_dt: Optional[datetime] = None

            snippet = ""
            sibling_p = a.find_next("p")
            if sibling_p:
                snippet = sibling_p.get_text(strip=True)[:600]

            items.append(
                {
                    "source": "Yahoo Finance",
                    "currency_pair": pair,
                    "title": title,
                    "url": href,
                    "published_dt": published_dt,
                    "snippet": snippet,
                }
            )

            if max_items is not None and len(items) >= max_items:
                break
        except Exception:
            continue

    print(f"[INFO] Yahoo Finance items collected: {len(items)}")
    return items


def collect_news_for_pair(
    pair: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[NewsItem]:
    """
    Aggregate news from all configured sources, de-duplicate, and convert
    to NewsItem objects.
    """
    raw_items: List[Dict] = []

    # Currently Investing.com and MarketWatch are used as sources.
    cfg = SUPPORTED_PAIRS.get(pair)
    if not cfg:
        raise SystemExit(
            f"Unsupported currency_pair {pair!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_PAIRS.keys()))}"
        )

    remaining = max_items

    # 1) Investing.com
    before = len(raw_items)
    raw_items.extend(
        scrape_investing_pair(
            pair=pair,
            url=cfg["investing_url"],
            start_date=start_date,
            end_date=end_date,
            max_items=remaining,
        )
    )
    if remaining is not None:
        added = len(raw_items) - before
        remaining = max(0, remaining - added)

    # 2) MarketWatch (only if we still want more items and URL is configured)
    mw_url = cfg.get("marketwatch_url")
    if mw_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_marketwatch_pair(
                pair=pair,
                url=mw_url,
                start_date=start_date,
                end_date=end_date,
                max_items=remaining,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 3) Yahoo Finance (currently configured only for some pairs, e.g. EUR/USD)
    yahoo_url = cfg.get("yahoo_url")
    if yahoo_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_yahoo_pair(
                pair=pair,
                url=yahoo_url,
                max_items=remaining,
            )
        )

    if not raw_items:
        return []

    # De-duplicate by (source, title, url)
    seen = set()
    unique_items: List[NewsItem] = []
    for idx, item in enumerate(raw_items, start=1):
        key = (item["source"], item["title"], item["url"])
        if key in seen:
            continue
        seen.add(key)

        published_dt = item.get("published_dt")
        published_at_str = (
            published_dt.astimezone(timezone.utc).isoformat()
            if isinstance(published_dt, datetime)
            else None
        )

        unique_items.append(
            NewsItem(
                id=len(unique_items) + 1,
                currency_pair=item["currency_pair"],
                source=item["source"],
                title=item["title"],
                url=item["url"],
                published_at=published_at_str,
                snippet=item.get("snippet", ""),
            )
        )

    print(f"\n[INFO] Total unique {pair} news items: {len(unique_items)}")
    return unique_items


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_news_to_json(news: List[NewsItem], path: str) -> None:
    payload = [asdict(item) for item in news]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved {len(news)} news items to {path!r}")


def load_news_from_json(path: str) -> List[NewsItem]:
    if not os.path.exists(path):
        debug_log(f"[WARN] JSON file {path!r} does not exist; returning empty list.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    news: List[NewsItem] = []
    for obj in raw:
        news.append(
            NewsItem(
                id=obj.get("id", len(news) + 1),
                currency_pair=obj.get("currency_pair", "UNKNOWN"),
                source=obj.get("source", "Unknown"),
                title=obj.get("title", ""),
                url=obj.get("url", ""),
                published_at=obj.get("published_at"),
                snippet=obj.get("snippet", ""),
            )
        )
    print(f"[INFO] Loaded {len(news)} news items from {path!r}")
    return news


def load_default_config() -> Dict[str, Optional[str]]:
    """
    Load default configuration from default_config.json if present.
    Only keys relevant to this script are used; unknown keys are ignored.
    """
    if not os.path.exists(DEFAULT_CONFIG_PATH):
        return {
            "start_date": None,
            "end_date": None,
            "currency_pair": "EUR/USD",
            "max_news": None,
            "enabled_llm_providers": ["deepseek", "openai"],
        }

    try:
        with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # If config is malformed, silently fall back to automatic date range
        return {
            "start_date": None,
            "end_date": None,
            "currency_pair": "EUR/USD",
            "max_news": None,
            "enabled_llm_providers": ["deepseek", "openai"],
        }

    start_date = data.get("start_date")
    end_date = data.get("end_date")
    currency_pair = data.get("currency_pair", "EUR/USD")
    max_news = data.get("max_news", None)
    if currency_pair not in SUPPORTED_PAIRS:
        # Fallback to EUR/USD if an unsupported pair is configured.
        currency_pair = "EUR/USD"

    # Parse llm_models: each model can have enabled=true/false
    enabled_providers: List[str] = []
    llm_models = data.get("llm_models")
    if isinstance(llm_models, dict):
        for name in ("deepseek", "openai", "gemini"):
            model_cfg = llm_models.get(name)
            if isinstance(model_cfg, dict) and model_cfg.get("enabled") is True:
                enabled_providers.append(name)
    if not enabled_providers:
        # Fallback: enable deepseek and openai if no valid config
        enabled_providers = ["deepseek", "openai"]

    return {
        "start_date": start_date if isinstance(start_date, str) else None,
        "end_date": end_date if isinstance(end_date, str) else None,
        "currency_pair": currency_pair,
        "max_news": max_news if isinstance(max_news, int) and max_news > 0 else None,
        "enabled_llm_providers": enabled_providers,
    }


# ---------------------------------------------------------------------------
# LLM integration
# ---------------------------------------------------------------------------

def build_news_context_for_llm(news: List[NewsItem],
                               max_articles: int = 8,
                               fetch_full_paragraphs: bool = True) -> str:
    """
    Build a compact, human-readable context for the LLM from the news list.
    Optionally re-fetch the article pages and replace the snippet with
    the first 1–2 paragraphs.
    """
    selected = news[:max_articles]
    blocks: List[str] = []
    for item in selected:
        snippet = item.snippet or ""
        if fetch_full_paragraphs and item.url:
            body = fetch_article_snippet(item.url, max_paragraphs=2)
            if body:
                snippet = body

        blocks.append(
            f"News {item.id}:\n"
            f"Source: {item.source}\n"
            f"Time: {item.published_at}\n"
            f"Title: {item.title}\n"
            f"Content: {snippet}\n"
            f"URL: {item.url}\n"
            f"---"
        )
    return "\n\n".join(blocks)


def build_analysis_prompt(news_context: str, pair: str) -> str:
    return (
        f"You are a professional {pair} forex analyst.\n"
        f"Based on the following recent {pair}-related news, provide:\n\n"
        "1. Key market drivers (economic data, monetary policy, risk sentiment).\n"
        f"2. Short-term (1–3 days) outlook for {pair}.\n"
        f"3. Medium-term (1–2 weeks) outlook for {pair}.\n"
        "4. Trading bias (bullish / bearish / sideways) with concise reasoning.\n"
        "5. Main risks or alternative scenarios that could invalidate your view.\n\n"
        "Make the answer structured, concise, and practical for a trader.\n\n"
        "Here is the news context:\n\n"
        f"{news_context}\n"
    )


def make_deepseek_client() -> Optional[OpenAI]:
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        debug_log(
            "[WARN] DEEPSEEK_API_KEY not set; DeepSeek analysis will be skipped."
        )
        return None
    return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")


def make_openai_client() -> Optional[OpenAI]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        debug_log(
            "[WARN] OPENAI_API_KEY not set; OpenAI analysis will be skipped."
        )
        return None
    # Default base_url is https://api.openai.com/v1
    return OpenAI(api_key=key)


def make_gemini_client() -> Optional[OpenAI]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        debug_log(
            "[WARN] GEMINI_API_KEY not set; Gemini analysis will be skipped."
        )
        return None
    return OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def run_llm_analysis(news: List[NewsItem],
                     pair: str,
                     enabled_providers: Optional[List[str]] = None,
                     max_articles: int = 8) -> Dict[str, str]:
    if not news:
        return {"error": "No news available for analysis."}
    if enabled_providers is None:
        enabled_providers = ["deepseek", "openai"]

    context = build_news_context_for_llm(
        news, max_articles=max_articles, fetch_full_paragraphs=True
    )
    prompt = build_analysis_prompt(context, pair)

    results: Dict[str, str] = {}
    providers = set(enabled_providers)

    if "deepseek" in providers:
        deepseek = make_deepseek_client()
        if deepseek is not None:
            print("\n[INFO] Requesting analysis from DeepSeek...")
            try:
                resp = deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                )
                results["deepseek"] = resp.choices[0].message.content.strip()
            except Exception as exc:
                results["deepseek"] = f"Error calling DeepSeek: {exc}"

    if "openai" in providers:
        client = make_openai_client()
        if client is not None:
            print("\n[INFO] Requesting analysis from OpenAI...")
            try:
                resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                )
                results["openai"] = resp.choices[0].message.content.strip()
            except Exception as exc:
                results["openai"] = f"Error calling OpenAI: {exc}"

    if "gemini" in providers:
        client = make_gemini_client()
        if client is not None:
            print("\n[INFO] Requesting analysis from Gemini...")
            try:
                resp = client.chat.completions.create(
                    model="gemini-2.0-flash",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=1500,
                )
                results["gemini"] = resp.choices[0].message.content.strip()
            except Exception as exc:
                results["gemini"] = f"Error calling Gemini: {exc}"

    if not results:
        results["error"] = "No LLM providers were configured or all calls failed."

    return results


def print_analysis_report(results: Dict[str, str], pair: str) -> None:
    print("\n" + "=" * 80)
    print(f" {pair} Forex News Analysis Report ")
    print("=" * 80)

    if not results:
        print("No analysis results available.")
        return

    ordered_keys = []
    for name in ("deepseek", "openai", "gemini"):
        if name in results:
            ordered_keys.append(name)
    for name in results.keys():
        if name not in ordered_keys:
            ordered_keys.append(name)

    for key in ordered_keys:
        print(f"\n[{key.upper()}]")
        print("-" * 80)
        print(results[key])


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FX news crawler and LLM-based analysis (Investing.com)."
    )
    parser.add_argument(
        "--start-date",
        "-s",
        help="Start date (inclusive) in YYYY-MM-DD. Default: 7 days ago.",
        default=None,
    )
    parser.add_argument(
        "--end-date",
        "-e",
        help="End date (inclusive) in YYYY-MM-DD. Default: today (UTC).",
        default=None,
    )
    parser.add_argument(
        "--max-articles",
        "-n",
        type=int,
        default=8,
        help="Maximum number of news items to send to the LLM.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Path to JSON file for storing crawled news. "
             "If omitted, a name based on the currency pair will be used.",
    )
    parser.add_argument(
        "--no-llm",
        dest="no_llm",
        action="store_true",
        help="Only crawl and save JSON; skip LLM analysis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (warnings, HTTP errors, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    # Load .env for API keys
    load_dotenv()

    args = parse_args()

    # Configure global verbosity flag
    global VERBOSE
    VERBOSE = bool(args.verbose)
    cfg = load_default_config()

    # Selected currency pair (from config only, to keep usage simple)
    pair = cfg.get("currency_pair", "EUR/USD")
    if pair not in SUPPORTED_PAIRS:
        pair = "EUR/USD"
    print(f"[INFO] Selected currency_pair from config: {pair}")

    # Date window precedence:
    # 1) CLI args if provided
    # 2) default_config.json if valid
    # 3) Fallback: last 7 days
    start_date = parse_cli_date(args.start_date or cfg.get("start_date"))
    end_date = parse_cli_date(args.end_date or cfg.get("end_date"))
    if not start_date and not end_date:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=7)
        print(
            f"[INFO] No date range provided. Using last 7 days: "
            f"{start_date.isoformat()} to {end_date.isoformat()} (UTC)."
        )
    elif start_date and end_date:
        print(
            f"[INFO] Using configured date range: "
            f"{start_date.isoformat()} to {end_date.isoformat()} (UTC)."
        )

    max_news = cfg.get("max_news")

    news_items = collect_news_for_pair(pair, start_date, end_date, max_items=max_news)
    if not news_items:
        print("[INFO] No news items collected; nothing to analyze.")
        return

    # If no explicit output path is provided, derive one from the pair,
    # e.g. "EUR/USD" -> "eur_usd_news.json"
    if args.output:
        output_path = args.output
    else:
        safe_pair = pair.replace("/", "_").replace(" ", "").lower()
        output_path = f"{safe_pair}_news.json"

    save_news_to_json(news_items, output_path)

    if args.no_llm:
        print("[INFO] Skipping LLM analysis (--no-llm specified).")
        return

    results = run_llm_analysis(
        news_items,
        pair=pair,
        enabled_providers=cfg.get("enabled_llm_providers", ["deepseek", "openai"]),
        max_articles=args.max_articles,
    )
    print_analysis_report(results, pair)


if __name__ == "__main__":
    main()