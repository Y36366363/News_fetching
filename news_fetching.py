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
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from dotenv import load_dotenv
from openai import OpenAI

from asset_urls import SUPPORTED_PAIRS, SUPPORTED_STOCKS


# Configuration

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Investing.com (especially equities pages) can be protected by Cloudflare.
# A slightly more browser-like header set often avoids the JS challenge page.
INVESTING_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
        "image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": "\"Chromium\";v=\"122\", \"Not(A:Brand\";v=\"24\", \"Google Chrome\";v=\"122\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"macOS\"",
    "Referer": "https://www.investing.com/",
}

# Path to default configuration (date range etc.), next to this script.
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

# Global flag to control verbose output (warnings, low-level errors).
VERBOSE = False


@dataclass
class NewsItem:
    id: int
    currency_pair: str  # For backward compatibility, also used for stock symbol
    source: str
    title: str
    url: str
    published_at: Optional[str]  # ISO 8601 string or None
    snippet: str                 # short content or first 1–2 paragraphs
    asset_type: str = "currency"  # "currency" or "stock"


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
            headers = INVESTING_HEADERS if "investing.com" in url else HEADERS
            resp = requests.get(url, headers=headers, timeout=15)
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


def fetch_article_full_text(url: str) -> str:
    """
    Best-effort extraction of an article's full readable text from its URL.
    """
    html = fetch_html(url)
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    article = soup.find("article")
    main = soup.find("main")
    if article is not None:
        candidates.append(article)
    if main is not None and main is not article:
        candidates.append(main)
    if soup.body is not None and soup.body is not article and soup.body is not main:
        candidates.append(soup.body)
    candidates.append(soup)

    def extract_from(node) -> str:
        paras: List[str] = []
        for p in node.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if not txt or len(txt) < 40:
                continue
            paras.append(txt)
        return "\n\n".join(paras)

    # Prefer roots that look like they contain real article body.
    for node in candidates:
        text = extract_from(node)
        if len(text) >= 800:  # enough substance to be a real article
            return text
        # If it has at least a few solid paragraphs, accept it.
        if text.count("\n\n") >= 3 and len(text) >= 300:
            return text

    # Fall back to whatever we got from the full document.
    return extract_from(soup)


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
                    "asset_type": "currency",
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
                    "asset_type": "currency",
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
                    "asset_type": "currency",
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


# ---------------------------------------------------------------------------
# NewsAPI (API source)
# ---------------------------------------------------------------------------

_NEWSAPI_FULL_CACHE: Dict[str, Dict] = {}


def _make_newsapi_key() -> Optional[str]:
    key = os.getenv("NEWS_API_KEY")
    if not key:
        debug_log("[WARN] NEWS_API_KEY not set; NewsAPI source will be skipped.")
        return None
    return key


def _newsapi_query_for_fx_pair(pair: str) -> str:
    """
    NewsAPI matching is sensitive to the exact terms used in articles.
    Many FX articles do NOT write "USD/CNY" literally; they use names like
    "yuan/renminbi", or tickers like "USDCNY"/"USDJPY".
    """
    parts = pair.replace(" ", "").upper().split("/")
    base = parts[0] if len(parts) > 0 else pair.upper()
    quote = parts[1] if len(parts) > 1 else ""

    code_to_names = {
        "USD": ["dollar", "U.S. dollar", "US dollar"],
        "EUR": ["euro"],
        "JPY": ["yen"],
        "GBP": ["pound", "sterling"],
        "CNY": ["yuan", "renminbi"],
        "CNH": ["yuan", "renminbi"],
        "CAD": ["Canadian dollar", "loonie"],
        "AUD": ["Australian dollar", "aussie"],
    }

    pair_ticker = f"{base}{quote}" if quote else base

    base_names = code_to_names.get(base, [])
    quote_names = code_to_names.get(quote, []) if quote else []

    # Pair identifiers (codes + common tickers)
    pair_terms = [pair, pair.replace("/", " "), pair_ticker]

    # Extra terms for some pairs
    extra_terms: List[str] = []
    if pair_ticker == "USDCNY":
        extra_terms += ["CNH", "offshore yuan", "onshore yuan"]
    if pair_ticker == "USDJPY":
        extra_terms += ["Japan yen"]
    if pair_ticker == "GBPUSD":
        extra_terms += ["cable"]

    # Combine terms
    or_terms: List[str] = []
    or_terms += [f'"{t}"' for t in pair_terms if t]
    or_terms += [f'"{t}"' for t in extra_terms if t]

    # Add name-based conjunctions: (euro AND dollar), (yuan AND dollar), ...
    name_conjunctions: List[str] = []
    for bn in base_names:
        for qn in quote_names:
            name_conjunctions.append(f'("{bn}" AND "{qn}")')
    for c in name_conjunctions:
        or_terms.append(c)

    pair_block = " OR ".join(or_terms) if or_terms else f'"{pair}"'

    # FX intent block (keep reasonably broad)
    intent_block = (
        'forex OR fx OR currency OR "exchange rate" OR "central bank" '
        'OR "interest rate" OR "rate cut" OR "rate hike"'
    )

    return f"({pair_block}) AND ({intent_block})"


def fetch_newsapi_fx_pair(
    pair: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Fetch FX-related news from NewsAPI.org (/v2/everything).
    Returns items compatible with the rest of the pipeline.
    """
    key = _make_newsapi_key()
    if not key:
        return []

    q = _newsapi_query_for_fx_pair(pair)

    page_size = 30
    if isinstance(max_items, int) and max_items > 0:
        page_size = min(max_items, 100)

    params = {
        "q": q,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": key,
    }
    if start_date:
        params["from"] = start_date.isoformat()
    if end_date:
        # NewsAPI expects ISO8601; date is accepted as YYYY-MM-DD
        params["to"] = end_date.isoformat()

    print(f"\n[INFO] Fetching NewsAPI.org ({pair} news)...")
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            headers=HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        debug_log(f"[WARN] NewsAPI request failed: {exc}")
        return []

    articles = data.get("articles") if isinstance(data, dict) else None
    if not isinstance(articles, list):
        return []

    # Cache full NewsAPI articles for this run so we can optionally write
    # a separate "full content" JSON file without making a second API call.
    # (Never store apiKey.)
    try:
        cached_params = {k: v for k, v in params.items() if k != "apiKey"}
        full_articles: List[Dict] = []
        for a in articles:
            if not isinstance(a, dict):
                continue
            published_at = a.get("publishedAt")
            published_dt = (
                parse_iso_like_datetime(published_at)
                if isinstance(published_at, str) and published_at.strip()
                else None
            )
            if not within_date_range(published_dt, start_date, end_date):
                continue
            full_articles.append(
                {
                    "source": a.get("source"),
                    "author": a.get("author"),
                    "title": a.get("title"),
                    "description": a.get("description"),
                    "content": a.get("content"),
                    "url": a.get("url"),
                    "urlToImage": a.get("urlToImage"),
                    "publishedAt": (
                        published_dt.astimezone(timezone.utc).isoformat()
                        if isinstance(published_dt, datetime)
                        else published_at
                    ),
                }
            )

        _NEWSAPI_FULL_CACHE[pair] = {
            "schema_version": 1,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "request": {
                "endpoint": "https://newsapi.org/v2/everything",
                "params": cached_params,
            },
            "response": {
                "status": data.get("status"),
                "totalResults": data.get("totalResults"),
            },
            "articles": full_articles,
        }
    except Exception:
        # Cache is best-effort; never fail the run because of it.
        pass

    items: List[Dict] = []
    for a in articles:
        try:
            title = (a.get("title") or "").strip()
            url = (a.get("url") or "").strip()
            if not title or not url:
                continue

            published_dt: Optional[datetime] = None
            published_at = a.get("publishedAt")
            if isinstance(published_at, str) and published_at.strip():
                published_dt = parse_iso_like_datetime(published_at)

            if not within_date_range(published_dt, start_date, end_date):
                continue

            snippet = (
                (a.get("description") or "")
                or (a.get("content") or "")
                or ""
            ).strip()

            src = a.get("source") or {}
            src_name = src.get("name") if isinstance(src, dict) else None
            source_label = f"NewsAPI: {src_name}" if src_name else "NewsAPI"

            items.append(
                {
                    "source": source_label,
                    "currency_pair": pair,
                    "asset_type": "currency",
                    "title": title,
                    "url": url,
                    "published_dt": published_dt,
                    "snippet": snippet[:600],
                }
            )

            if max_items is not None and len(items) >= max_items:
                break
        except Exception:
            continue

    print(f"[INFO] NewsAPI.org items collected: {len(items)}")
    return items


def save_newsapi_full_json(pair: str, path: str) -> bool:
    """
    Save a separate JSON containing the full NewsAPI articles (including `content`)
    for later reuse.
    """
    payload = _NEWSAPI_FULL_CACHE.get(pair)
    if not isinstance(payload, dict):
        return False

    # Expand truncated `content` like "... [+1234 chars]" by fetching the
    # original article URL and extracting full text. This is best-effort:
    # some sites may block scraping or show paywalls.
    try:
        articles = payload.get("articles")
        if isinstance(articles, list):
            for a in articles:
                if not isinstance(a, dict):
                    continue
                content = a.get("content")
                url = a.get("url")
                if not (isinstance(content, str) and isinstance(url, str) and url):
                    continue

                # NewsAPI often truncates content and appends "[+NNNN chars]".
                if not re.search(r"\[\+\d+\schars\]\s*$", content):
                    continue

                full_text = fetch_article_full_text(url)
                if full_text:
                    a["content_truncated"] = content
                    a["content"] = full_text
                    a["content_extracted_from_url"] = True
                else:
                    a["content_extracted_from_url"] = False
    except Exception:
        pass

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved NewsAPI full content to {path!r}")
        return True
    except Exception as exc:
        debug_log(f"[WARN] Failed to save NewsAPI full JSON: {exc}")
        return False


# ---------------------------------------------------------------------------
# Stock scrapers
# ---------------------------------------------------------------------------

def scrape_investing_stock(
    symbol: str,
    url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a stock's news from Investing.com.
    Similar to scrape_investing_pair but for stocks.
    """
    print(f"\n[INFO] Crawling Investing.com ({symbol} news)...")
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
                href = "https://www.investing.com" + href

            # Filter out section navigation links; prefer article-like URLs.
            # Many Investing article URLs end with a numeric id (e.g. -4501054).
            import re
            if not re.search(r"-\d{4,}($|\?)", href):
                continue

            published_dt: Optional[datetime] = None
            parent = a.parent
            if parent is not None:
                time_tag = parent.find("time")
                if time_tag and time_tag.get_text(strip=True):
                    published_dt = parse_iso_like_datetime(time_tag.get_text(strip=True))

            if not within_date_range(published_dt, start_date, end_date):
                continue

            snippet = ""
            sibling_p = a.find_next("p")
            if sibling_p:
                snippet = sibling_p.get_text(strip=True)[:600]

            items.append(
                {
                    "source": "Investing.com",
                    "currency_pair": symbol,  # Reusing field name for stock symbol
                    "asset_type": "stock",
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


def scrape_marketwatch_stock(
    symbol: str,
    url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a stock's news section on MarketWatch.
    """
    print(f"\n[INFO] Crawling MarketWatch ({symbol} news)...")
    html = fetch_html(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    items: List[Dict] = []

    for a in soup.select("a[href*='mod=mw_quote_news'], a[href*='/story/']"):
        try:
            title = a.get_text(strip=True)
            if not title or len(title) < 15:
                continue

            href = a.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = "https://www.marketwatch.com" + href

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
                    "currency_pair": symbol,
                    "asset_type": "stock",
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


def scrape_yahoo_stock(
    symbol: str,
    url: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
) -> List[Dict]:
    """
    Best-effort scraper for a stock's quote page on Yahoo Finance.
    """
    print(f"\n[INFO] Crawling Yahoo Finance ({symbol} quote news)...")
    # Yahoo Finance quote pages often load the story list via JS, so the HTML
    # may not contain real headline anchors. Use the public RSS feed as the
    # primary, stable source.
    rss_url = (
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?"
        f"s={symbol}&region=US&lang=en-US"
    )

    xml_text = fetch_html(rss_url)
    items: List[Dict] = []
    if xml_text:
        try:
            root = ET.fromstring(xml_text)
            for it in root.findall(".//item"):
                title = (it.findtext("title") or "").strip()
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                desc = (it.findtext("description") or "").strip()

                if not title or not link:
                    continue

                published_dt: Optional[datetime] = None
                if pub:
                    try:
                        published_dt = date_parser.parse(pub)
                        if published_dt.tzinfo is None:
                            published_dt = published_dt.replace(tzinfo=timezone.utc)
                        else:
                            published_dt = published_dt.astimezone(timezone.utc)
                    except Exception:
                        published_dt = None

                if not within_date_range(published_dt, start_date, end_date):
                    continue

                snippet = ""
                if desc:
                    # RSS descriptions can contain HTML.
                    snippet = BeautifulSoup(desc, "html.parser").get_text(
                        " ", strip=True
                    )[:600]

                items.append(
                    {
                        "source": "Yahoo Finance",
                        "currency_pair": symbol,
                        "asset_type": "stock",
                        "title": title,
                        "url": link,
                        "published_dt": published_dt,
                        "snippet": snippet,
                    }
                )

                if max_items is not None and len(items) >= max_items:
                    break
        except Exception:
            items = []

    # Fallback: attempt best-effort HTML scrape if RSS fails for any reason.
    if not items:
        html = fetch_html(url)
        if not html:
            print("[INFO] Yahoo Finance items collected: 0")
            return []

        soup = BeautifulSoup(html, "html.parser")
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

                items.append(
                    {
                        "source": "Yahoo Finance",
                        "currency_pair": symbol,
                        "asset_type": "stock",
                        "title": title,
                        "url": href,
                        "published_dt": None,
                        "snippet": "",
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
    enable_investing: bool = True,
    enable_marketwatch: bool = True,
    enable_yahoo: bool = True,
    enable_newsapi: bool = False,
    newsapi_max_items: Optional[int] = None,
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
    if enable_investing and (remaining is None or remaining > 0):
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
    if enable_marketwatch and mw_url and (remaining is None or remaining > 0):
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
    if enable_yahoo and yahoo_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_yahoo_pair(
                pair=pair,
                url=yahoo_url,
                max_items=remaining,
            )
        )

    # 4) NewsAPI.org (optional API source)
    # If enabled, always call it so we can also save a separate "full content"
    # NewsAPI JSON file for future use (even if max_items was already reached).
    if enable_newsapi:
        cap = newsapi_max_items if isinstance(newsapi_max_items, int) else None
        newsapi_items = fetch_newsapi_fx_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_items=cap,
        )

        if remaining is None or remaining > 0:
            before = len(raw_items)
            if remaining is None:
                raw_items.extend(newsapi_items)
            else:
                raw_items.extend(newsapi_items[:remaining])
            added = len(raw_items) - before
            if remaining is not None:
                remaining = max(0, remaining - added)

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
                asset_type=item.get("asset_type", "currency"),
            )
        )

    print(f"\n[INFO] Total unique {pair} news items: {len(unique_items)}")
    return unique_items


def collect_news_for_stock(
    symbol: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: Optional[int] = None,
    enable_investing: bool = True,
    enable_marketwatch: bool = True,
    enable_yahoo: bool = True,
) -> List[NewsItem]:
    """
    Aggregate news from all configured sources for a stock, de-duplicate, and convert
    to NewsItem objects.
    """
    raw_items: List[Dict] = []

    cfg = SUPPORTED_STOCKS.get(symbol)
    if not cfg:
        raise SystemExit(
            f"Unsupported stock {symbol!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_STOCKS.keys()))}"
        )

    remaining = max_items

    # 1) Investing.com
    investing_url = cfg.get("investing_url")
    if enable_investing and investing_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_investing_stock(
                symbol=symbol,
                url=investing_url,
                start_date=start_date,
                end_date=end_date,
                max_items=remaining,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 2) MarketWatch
    mw_url = cfg.get("marketwatch_url")
    if enable_marketwatch and mw_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_marketwatch_stock(
                symbol=symbol,
                url=mw_url,
                start_date=start_date,
                end_date=end_date,
                max_items=remaining,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 3) Yahoo Finance
    yahoo_url = cfg.get("yahoo_url")
    if enable_yahoo and yahoo_url and (remaining is None or remaining > 0):
        before = len(raw_items)
        raw_items.extend(
            scrape_yahoo_stock(
                symbol=symbol,
                url=yahoo_url,
                start_date=start_date,
                end_date=end_date,
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
                asset_type=item.get("asset_type", "stock"),
            )
        )

    print(f"\n[INFO] Total unique {symbol} news items: {len(unique_items)}")
    return unique_items


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_news_to_json(
    news: List[NewsItem],
    path: str,
    *,
    meta: Optional[Dict] = None,
    analysis: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save crawled news to JSON.

    Backward compatible:
    - If meta/analysis are not provided, writes a simple list (legacy format).
    - If meta and/or analysis are provided, writes a report object:
      {schema_version, meta, news, analysis}
    """
    payload_news = [asdict(item) for item in news]
    if meta is None and analysis is None:
        payload = payload_news
    else:
        payload = {
            "schema_version": 2,
            "meta": meta or {},
            "news": payload_news,
            "analysis": analysis or {},
        }
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

    # Support both legacy list format and the newer report format.
    if isinstance(raw, dict) and isinstance(raw.get("news"), list):
        raw_items = raw["news"]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raw_items = []

    for obj in raw_items:
        news.append(
            NewsItem(
                id=obj.get("id", len(news) + 1),
                currency_pair=obj.get("currency_pair", "UNKNOWN"),
                source=obj.get("source", "Unknown"),
                title=obj.get("title", ""),
                url=obj.get("url", ""),
                published_at=obj.get("published_at"),
                snippet=obj.get("snippet", ""),
                asset_type=obj.get("asset_type", "currency"),
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
            "asset_type": "currency",
            "currency_pair": "EUR/USD",
            "stock": None,
            "max_news": None,
            "enable_investing": True,
            "enable_marketwatch": True,
            "enable_yahoo": True,
            "enable_newsapi": False,
            "newsapi_max_items": None,
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
            "asset_type": "currency",
            "currency_pair": "EUR/USD",
            "stock": None,
            "max_news": None,
            "enable_investing": True,
            "enable_marketwatch": True,
            "enable_yahoo": True,
            "enable_newsapi": False,
            "newsapi_max_items": None,
            "enabled_llm_providers": ["deepseek", "openai"],
        }

    start_date = data.get("start_date")
    end_date = data.get("end_date")
    asset_type = data.get("asset_type", "currency")
    if asset_type not in ("currency", "stock"):
        asset_type = "currency"
    
    currency_pair = data.get("currency_pair", "EUR/USD")
    stock = data.get("stock")
    max_news = data.get("max_news", None)
    enable_investing = bool(data.get("enable_investing", True))
    enable_marketwatch = bool(data.get("enable_marketwatch", True))
    enable_yahoo = bool(data.get("enable_yahoo", True))
    enable_newsapi = bool(data.get("enable_newsapi", False))
    newsapi_max_items = data.get("newsapi_max_items", None)
    if not (isinstance(newsapi_max_items, int) and newsapi_max_items > 0):
        newsapi_max_items = None
    
    if asset_type == "currency":
        if currency_pair not in SUPPORTED_PAIRS:
            currency_pair = "EUR/USD"
    elif asset_type == "stock":
        if not stock or stock not in SUPPORTED_STOCKS:
            stock = "AAPL"  # Default to Apple if invalid stock

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
        "asset_type": asset_type,
        "currency_pair": currency_pair,
        "stock": stock if isinstance(stock, str) else None,
        "max_news": max_news if isinstance(max_news, int) and max_news > 0 else None,
        "enable_investing": enable_investing,
        "enable_marketwatch": enable_marketwatch,
        "enable_yahoo": enable_yahoo,
        "enable_newsapi": enable_newsapi,
        "newsapi_max_items": newsapi_max_items,
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


def build_analysis_prompt(news_context: str, asset: str, asset_type: str = "currency") -> str:
    """
    Build analysis prompt for either currency pairs or stocks.
    """
    if asset_type == "stock":
        return (
            f"You are a professional {asset} stock analyst.\n"
            f"Based on the following recent {asset}-related news, provide:\n\n"
            "1. Key market drivers (earnings, product launches, market trends, sector dynamics).\n"
            f"2. Short-term (1–3 days) outlook for {asset} stock price.\n"
            f"3. Medium-term (1–2 weeks) outlook for {asset} stock price.\n"
            "4. Trading bias (bullish / bearish / sideways) with concise reasoning.\n"
            "5. Main risks or alternative scenarios that could invalidate your view.\n\n"
            "Make the answer structured, concise, and practical for a trader.\n\n"
            "Here is the news context:\n\n"
            f"{news_context}\n"
        )
    else:
        return (
            f"You are a professional {asset} forex analyst.\n"
            f"Based on the following recent {asset}-related news, provide:\n\n"
            "1. Key market drivers (economic data, monetary policy, risk sentiment).\n"
            f"2. Short-term (1–3 days) outlook for {asset}.\n"
            f"3. Medium-term (1–2 weeks) outlook for {asset}.\n"
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
                     asset: str,
                     asset_type: str = "currency",
                     enabled_providers: Optional[List[str]] = None,
                     max_articles: int = 8) -> Dict[str, str]:
    if not news:
        return {"error": "No news available for analysis."}
    if enabled_providers is None:
        enabled_providers = ["deepseek", "openai"]

    context = build_news_context_for_llm(
        news, max_articles=max_articles, fetch_full_paragraphs=True
    )
    prompt = build_analysis_prompt(context, asset, asset_type)

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


def print_analysis_report(results: Dict[str, str], asset: str, asset_type: str = "currency") -> None:
    asset_label = f"{asset} Stock" if asset_type == "stock" else f"{asset} Forex"
    print("\n" + "=" * 80)
    print(f" {asset_label} News Analysis Report ")
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

    # Determine asset type and selection
    asset_type = cfg.get("asset_type", "currency")
    if asset_type == "stock":
        stock = cfg.get("stock", "AAPL")
        if stock not in SUPPORTED_STOCKS:
            stock = "AAPL"
        print(f"[INFO] Selected stock from config: {stock}")
        asset = stock
    else:
        pair = cfg.get("currency_pair", "EUR/USD")
        if pair not in SUPPORTED_PAIRS:
            pair = "EUR/USD"
        print(f"[INFO] Selected currency_pair from config: {pair}")
        asset = pair

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

    # Collect news based on asset type
    if asset_type == "stock":
        news_items = collect_news_for_stock(
            asset,
            start_date,
            end_date,
            max_items=max_news,
            enable_investing=bool(cfg.get("enable_investing", True)),
            enable_marketwatch=bool(cfg.get("enable_marketwatch", True)),
            enable_yahoo=bool(cfg.get("enable_yahoo", True)),
        )
    else:
        news_items = collect_news_for_pair(
            asset,
            start_date,
            end_date,
            max_items=max_news,
            enable_investing=bool(cfg.get("enable_investing", True)),
            enable_marketwatch=bool(cfg.get("enable_marketwatch", True)),
            enable_yahoo=bool(cfg.get("enable_yahoo", True)),
            enable_newsapi=bool(cfg.get("enable_newsapi", False)),
            newsapi_max_items=cfg.get("newsapi_max_items", None),
        )

    if not news_items:
        print("[INFO] No news items collected; nothing to analyze.")
        return

    # If no explicit output path is provided, derive one from the asset
    if args.output:
        output_path = args.output
    else:
        if asset_type == "stock":
            safe_asset = asset.lower()
            output_path = f"{safe_asset}_news.json"
        else:
            safe_asset = asset.replace("/", "_").replace(" ", "").lower()
            output_path = f"{safe_asset}_news.json"

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset_type": asset_type,
        "asset": asset,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "max_news": max_news,
        "max_articles": args.max_articles,
        "sources": {
            "investing": bool(cfg.get("enable_investing", True)),
            "marketwatch": bool(cfg.get("enable_marketwatch", True)),
            "yahoo": bool(cfg.get("enable_yahoo", True)),
            "newsapi": bool(cfg.get("enable_newsapi", False)),
        },
        "newsapi_max_items": cfg.get("newsapi_max_items", None),
        "llm_enabled": cfg.get("enabled_llm_providers", []),
    }

    # Additionally save a separate NewsAPI-only JSON with full content
    # for future access/reuse.
    if (
        asset_type == "currency"
        and bool(cfg.get("enable_newsapi", False))
        and isinstance(asset, str)
    ):
        base_no_ext, _ = os.path.splitext(output_path)
        if base_no_ext.endswith("_news"):
            newsapi_full_path = base_no_ext[:-5] + "_newsapi_full.json"
        else:
            newsapi_full_path = base_no_ext + "_newsapi_full.json"
        save_newsapi_full_json(asset, newsapi_full_path)

    if args.no_llm:
        save_news_to_json(news_items, output_path, meta=meta, analysis={})
        print("[INFO] Skipping LLM analysis (--no-llm specified).")
        return

    results = run_llm_analysis(
        news_items,
        asset=asset,
        asset_type=asset_type,
        enabled_providers=cfg.get("enabled_llm_providers", ["deepseek", "openai"]),
        max_articles=args.max_articles,
    )
    save_news_to_json(news_items, output_path, meta=meta, analysis=results)
    print_analysis_report(results, asset, asset_type)


if __name__ == "__main__":
    main()