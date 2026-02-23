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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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


def normalize_fx_pair(value: object) -> str:
    """
    Normalize common FX pair inputs into the canonical form used by this project:
    - "AUD/CAD" -> "AUD/CAD"
    - "AUD_CAD" -> "AUD/CAD"
    - "aud cad" -> "AUD/CAD"

    If it doesn't look like a 3-letter/3-letter pair, returns the trimmed string.
    """
    if not isinstance(value, str):
        return "EUR/USD"
    text = value.strip()
    m = re.match(r"^([A-Za-z]{3})\s*[_/ ]\s*([A-Za-z]{3})$", text)
    if m:
        return f"{m.group(1).upper()}/{m.group(2).upper()}"
    return text


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


# ---------------------------------------------------------------------------
# Alpha Vantage (API source)
# ---------------------------------------------------------------------------

_ALPHAVANTAGE_FULL_CACHE: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# EODHD (API source)
# ---------------------------------------------------------------------------

_EODHD_FULL_CACHE: Dict[str, Dict[str, Any]] = {}


def _make_eodhd_key() -> Optional[str]:
    key = (
        os.getenv("EODHD_API_KEY")
        or os.getenv("EODHD_API_TOKEN")
        or os.getenv("EODHD_TOKEN")
        or os.getenv("EODHD_KEY")
    )
    if not key:
        debug_log("[WARN] EODHD_API_KEY not set; EODHD source will be skipped.")
        return None
    return key


def _eodhd_symbol_for_fx_pair(pair: str) -> str:
    parts = pair.replace(" ", "").upper().split("/")
    base = parts[0] if len(parts) > 0 else pair.upper()
    quote = parts[1] if len(parts) > 1 else ""
    return f"{base}{quote}.FOREX" if quote else f"{base}.FOREX"


def _eodhd_symbol_for_stock(symbol: str) -> str:
    # Your NASDAQ-100 list is US tickers; EODHD uses suffixes like .US
    return f"{symbol.upper()}.US"


def _hostname_from_url(url: str) -> str:
    try:
        host = urlparse(url).netloc or ""
        return host.lower()
    except Exception:
        return ""


def fetch_eodhd_news(
    *,
    asset_label: str,
    eodhd_symbol: str,
    start_date: Optional[date],
    end_date: Optional[date],
    limit: int = 10,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Fetch financial news from EODHD News API: https://eodhd.com/api/news
    Returns normalized items compatible with the rest of the pipeline.
    """
    key = _make_eodhd_key()
    if not key:
        return []

    lim = limit if isinstance(limit, int) and 0 < limit <= 1000 else 10
    off = offset if isinstance(offset, int) and offset >= 0 else 0

    params: Dict[str, Any] = {
        "s": eodhd_symbol,
        "offset": off,
        "limit": lim,
        "api_token": key,
        "fmt": "json",
    }
    if start_date:
        params["from"] = start_date.isoformat()
    if end_date:
        params["to"] = end_date.isoformat()

    print(f"\n[INFO] Fetching EODHD news ({eodhd_symbol})...")
    try:
        resp = requests.get(
            "https://eodhd.com/api/news",
            params=params,
            headers=HEADERS,
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        debug_log(f"[WARN] EODHD request failed: {exc}")
        return []

    # EODHD returns list for success; dict for errors.
    if isinstance(data, dict):
        msg = data.get("message") or data.get("error") or data.get("Error Message")
        if isinstance(msg, str) and msg.strip():
            print(f"[WARN] EODHD returned: {msg.strip()}")
        else:
            print("[WARN] EODHD returned an unexpected response.")
        return []

    if not isinstance(data, list):
        return []

    # Cache full payload (without api_token).
    try:
        cached_params = {k: v for k, v in params.items() if k != "api_token"}
        _EODHD_FULL_CACHE[asset_label] = {
            "schema_version": 1,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "asset": asset_label,
            "symbol": eodhd_symbol,
            "request": {"endpoint": "https://eodhd.com/api/news", "params": cached_params},
            "articles": data,
        }
    except Exception:
        pass

    items: List[Dict[str, Any]] = []
    for a in data:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        link = (a.get("link") or "").strip()
        if not title or not link:
            continue

        published_dt: Optional[datetime] = None
        d = a.get("date")
        if isinstance(d, str) and d.strip():
            published_dt = parse_iso_like_datetime(d) or None

        if not within_date_range(published_dt, start_date, end_date):
            continue

        content = (a.get("content") or "").strip()
        snippet = content[:600] if content else ""

        host = _hostname_from_url(link)
        source_label = f"EODHD: {host}" if host else "EODHD"

        items.append(
            {
                "source": source_label,
                "currency_pair": asset_label,
                "asset_type": "currency",
                "title": title,
                "url": link,
                "published_dt": published_dt,
                "snippet": snippet,
            }
        )

        if len(items) >= lim:
            break

    print(f"[INFO] EODHD items collected: {len(items)}")
    return items


def _make_alphavantage_key() -> Optional[str]:
    key = (
        os.getenv("ALPHA_VANTAGE_KEY")
        or os.getenv("ALPHAVANTAGE_API_KEY")
        or os.getenv("ALPHA_VANTAGE_API_KEY")
    )
    if not key:
        debug_log("[WARN] ALPHA_VANTAGE_KEY not set; Alpha Vantage source will be skipped.")
        return None
    return key


def _alphavantage_time_from(d: Optional[date]) -> Optional[str]:
    if not isinstance(d, date):
        return None
    return d.strftime("%Y%m%dT0000")


def _alphavantage_time_to(d: Optional[date]) -> Optional[str]:
    if not isinstance(d, date):
        return None
    return d.strftime("%Y%m%dT2359")


def parse_alphavantage_time_published(value: str) -> Optional[datetime]:
    """
    Alpha Vantage NEWS_SENTIMENT returns time like YYYYMMDDTHHMMSS or YYYYMMDDTHHMM.
    Interpret it as UTC.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


def fetch_alphavantage_news_sentiment(
    *,
    tickers: Optional[str] = None,
    start_date: Optional[date],
    end_date: Optional[date],
    limit: int = 10,
    sort: str = "RELEVANCE",
    topics: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch financial news from Alpha Vantage NEWS_SENTIMENT.
    Returns normalized items compatible with the rest of the pipeline.
    """
    key = _make_alphavantage_key()
    if not key:
        return []

    lim = limit if isinstance(limit, int) and 0 < limit <= 1000 else 10
    params: Dict[str, Any] = {
        "function": "NEWS_SENTIMENT",
        "sort": sort,
        "limit": lim,
        "apikey": key,
    }
    if tickers:
        params["tickers"] = tickers
    if topics:
        params["topics"] = topics
    tf = _alphavantage_time_from(start_date)
    tt = _alphavantage_time_to(end_date)
    if tf:
        params["time_from"] = tf
    if tt:
        params["time_to"] = tt

    label = tickers or (f"topics={topics}" if topics else "all")
    print(f"\n[INFO] Fetching Alpha Vantage news ({label})...")
    def _do_request() -> Optional[Dict[str, Any]]:
        try:
            resp = requests.get(
                "https://www.alphavantage.co/query",
                params=params,
                headers=HEADERS,
                timeout=25,
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, dict) else None
        except Exception as exc:
            debug_log(f"[WARN] Alpha Vantage request failed: {exc}")
            return None

    data = _do_request()
    if data is None:
        return []

    # Alpha Vantage uses 200 responses with informational/error payloads.
    info = data.get("Information") or data.get("Note") or data.get("Error Message")
    if isinstance(info, str) and info.strip():
        msg = info.strip()
        # Free tier throttle is typically returned via "Note".
        # Respect 1 request/sec and retry once.
        if "spreading out your free API requests" in msg or "request per second" in msg:
            print(f"[WARN] Alpha Vantage rate-limited: {msg}")
            time.sleep(1.1)
            data2 = _do_request()
            if isinstance(data2, dict):
                info2 = data2.get("Information") or data2.get("Note") or data2.get("Error Message")
                if isinstance(info2, str) and info2.strip():
                    print(f"[WARN] Alpha Vantage returned: {info2.strip()}")
                    return []
                data = data2
            else:
                return []
        else:
            print(f"[WARN] Alpha Vantage returned: {msg}")
            return []

    # Cache full response for this run (without apikey).
    try:
        cached_params = {k: v for k, v in params.items() if k != "apikey"}
        cache_key = tickers or (f"topics:{topics}" if topics else "all")
        _ALPHAVANTAGE_FULL_CACHE[cache_key] = {
            "schema_version": 1,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "request": {"endpoint": "https://www.alphavantage.co/query", "params": cached_params},
            "response": {
                "items": data.get("items") if isinstance(data, dict) else None,
                "sentiment_score_definition": data.get("sentiment_score_definition") if isinstance(data, dict) else None,
                "relevance_score_definition": data.get("relevance_score_definition") if isinstance(data, dict) else None,
            },
            "raw": data,
        }
    except Exception:
        pass

    feed = data.get("feed") if isinstance(data, dict) else None
    if not isinstance(feed, list):
        if isinstance(data, dict) and data:
            # Surface unexpected shape to help debugging.
            print("[WARN] Alpha Vantage response did not contain a 'feed' list.")
        return []

    items: List[Dict[str, Any]] = []
    for a in feed:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        url = (a.get("url") or "").strip()
        if not title or not url:
            continue

        published_dt = None
        tp = a.get("time_published")
        if isinstance(tp, str):
            published_dt = parse_alphavantage_time_published(tp)
        if not within_date_range(published_dt, start_date, end_date):
            continue

        src_val = a.get("source") or a.get("source_domain") or "AlphaVantage"
        source = src_val.strip() if isinstance(src_val, str) else "AlphaVantage"
        summary = (a.get("summary") or a.get("description") or "").strip()
        if not summary:
            # As a fallback, stitch a short snippet from topics/sentiment.
            summary = ""

        items.append(
            {
                "source": f"AlphaVantage: {source}",
                # Caller should set currency_pair/asset_type appropriately.
                "currency_pair": "UNKNOWN",
                "asset_type": "currency",
                "title": title,
                "url": url,
                "published_dt": published_dt,
                "snippet": summary[:600],
            }
        )

        if len(items) >= lim:
            break

    print(f"[INFO] Alpha Vantage items collected: {len(items)}")
    return items


def fetch_alphavantage_fx_pair(
    pair: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    """
    Alpha Vantage tickers filter is AND-based. For FX pairs we use both currencies
    to prioritize pair-relevant news.
    """
    parts = pair.replace(" ", "").upper().split("/")
    base = parts[0] if len(parts) > 0 else pair.upper()
    quote = parts[1] if len(parts) > 1 else ""
    lim = max_items if isinstance(max_items, int) and max_items > 0 else 10

    # Avoid passing a future time_to (can cause empty results for some APIs).
    today = datetime.now(timezone.utc).date()
    effective_end = end_date if (isinstance(end_date, date) and end_date <= today) else None

    # Alpha Vantage NEWS_SENTIMENT ticker support for forex is inconsistent:
    # "FOREX:USD" is shown in the docs, but "FOREX:EUR" can return "Invalid inputs".
    # To make FX news reliable, we query by *topics* (single API call), then
    # filter locally for currency keywords.

    code_to_names = {
        "USD": ["dollar", "u.s. dollar", "us dollar"],
        "EUR": ["euro"],
        "JPY": ["yen"],
        "GBP": ["pound", "sterling"],
        "CNY": ["yuan", "renminbi"],
        "CNH": ["yuan", "renminbi"],
        "CAD": ["canadian dollar", "loonie"],
        "AUD": ["australian dollar", "aussie"],
    }
    base_names = code_to_names.get(base, [])
    quote_names = code_to_names.get(quote, []) if quote else []
    keyword_patterns: List[str] = []
    if base:
        keyword_patterns.append(rf"\\b{re.escape(base.lower())}\\b")
    if quote:
        keyword_patterns.append(rf"\\b{re.escape(quote.lower())}\\b")
    for n in base_names + quote_names:
        keyword_patterns.append(re.escape(n.lower()))

    topics = "economy_monetary,economy_macro,financial_markets"
    batch = fetch_alphavantage_news_sentiment(
        tickers=None,
        start_date=start_date,
        end_date=effective_end,
        limit=50,
        sort="LATEST",
        topics=topics,
    )

    filtered: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    for it in batch:
        url = it.get("url")
        if not (isinstance(url, str) and url) or url in seen_urls:
            continue
        title = (it.get("title") or "")
        snippet = (it.get("snippet") or "")
        text = f"{title} {snippet}".lower()
        if any(re.search(p, text) for p in keyword_patterns):
            it["currency_pair"] = pair
            it["asset_type"] = "currency"
            filtered.append(it)
            seen_urls.add(url)
            if len(filtered) >= lim:
                break

    print(f"[INFO] Alpha Vantage FX filtered items kept: {len(filtered)}")
    return filtered


def fetch_alphavantage_stock(
    symbol: str,
    start_date: Optional[date],
    end_date: Optional[date],
    max_items: int = 10,
) -> List[Dict[str, Any]]:
    lim = max_items if isinstance(max_items, int) and max_items > 0 else 10
    today = datetime.now(timezone.utc).date()
    effective_end = end_date if (isinstance(end_date, date) and end_date <= today) else None

    items = fetch_alphavantage_news_sentiment(
        tickers=symbol.upper(),
        start_date=start_date,
        end_date=effective_end,
        limit=lim,
        sort="RELEVANCE",
        topics="financial_markets,earnings,mergers_and_acquisitions,ipo",
    )
    for it in items:
        it["currency_pair"] = symbol
        it["asset_type"] = "stock"
    return items

def _load_json_file_best_effort(path: str) -> object:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _is_newsapi_content_truncated(text: str) -> bool:
    return bool(re.search(r"\[\+\d+\schars\]\s*$", text or ""))


def _normalize_single_line(text: str) -> str:
    """
    Collapse all whitespace (including newlines) into single spaces.
    Useful for storing long article content compactly in JSON.
    """
    if not isinstance(text, str):
        return ""
    # Split/join collapses runs of whitespace and removes \n/\r/\t.
    return " ".join(text.split()).strip()


def update_newsapi_content_store(pair: str, path: str) -> Dict[str, Any]:
    """
    Maintain an append-only NewsAPI *content* store:
    - Stores only article objects (root is a JSON list).
    - Does NOT overwrite/reset previous content each run; it only appends new items.
    - De-duplicates by `url`.

    Returns: {"total": int, "added": int, "skipped_existing": int, "path": str}
    """
    payload = _NEWSAPI_FULL_CACHE.get(pair)
    if not isinstance(payload, dict):
        return {"total": 0, "added": 0, "skipped_existing": 0, "path": path, "error": "No cached NewsAPI payload for this run."}

    existing_raw = _load_json_file_best_effort(path)
    existing_articles: List[Dict[str, Any]] = []
    if isinstance(existing_raw, list):
        existing_articles = [a for a in existing_raw if isinstance(a, dict)]
    elif isinstance(existing_raw, dict) and isinstance(existing_raw.get("articles"), list):
        # Backward compatibility: if an older dict format exists, read its articles.
        existing_articles = [a for a in existing_raw["articles"] if isinstance(a, dict)]

    # Normalize existing records in-place (single-line content; drop non-content metadata).
    url_to_existing: Dict[str, Dict[str, Any]] = {}
    for a in existing_articles:
        url = a.get("url")
        if not (isinstance(url, str) and url):
            continue
        a["content"] = _normalize_single_line(a.get("content") or "")
        # These are useful for debugging, but the user wants only main content.
        a.pop("content_truncated", None)
        a.pop("content_extracted_from_url", None)
        url_to_existing[url] = a

    existing_urls = set(url_to_existing.keys())

    cached_articles = payload.get("articles")
    if not isinstance(cached_articles, list):
        cached_articles = []

    added = 0
    skipped = 0
    upgraded_existing = 0
    new_articles: List[Dict[str, Any]] = []

    for a in cached_articles:
        if not isinstance(a, dict):
            continue
        url = a.get("url")
        if not (isinstance(url, str) and url):
            continue
        if url in existing_urls:
            # If we already have this URL but its content is still truncated/empty,
            # try to upgrade it to full extracted text (best-effort).
            existing = url_to_existing.get(url)
            if isinstance(existing, dict):
                existing_content = existing.get("content") or ""
                if (not isinstance(existing_content, str)) or _is_newsapi_content_truncated(existing_content) or len(existing_content) < 200:
                    full_text = fetch_article_full_text(url)
                    if full_text:
                        existing["content"] = _normalize_single_line(full_text)
                        upgraded_existing += 1
            skipped += 1
            continue

        # Expand truncated content best-effort for new items only.
        content = a.get("content")
        if isinstance(content, str) and _is_newsapi_content_truncated(content):
            full_text = fetch_article_full_text(url)
            if full_text:
                a["content"] = full_text
            else:
                # Keep truncated string as-is if we cannot fetch full text.
                pass

        # Keep only the fields we want in the content store.
        src = a.get("source")
        src_name = src.get("name") if isinstance(src, dict) else None
        stored: Dict[str, Any] = {
            "source": src_name or src or None,
            "author": a.get("author"),
            "title": a.get("title"),
            "description": a.get("description"),
            "url": url,
            "urlToImage": a.get("urlToImage"),
            "publishedAt": a.get("publishedAt"),
            "content": _normalize_single_line(a.get("content") or ""),
        }

        new_articles.append(stored)
        existing_urls.add(url)
        url_to_existing[url] = stored
        added += 1

    merged = existing_articles + new_articles
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(
            f"[INFO] Updated NewsAPI content store: added {added}, upgraded {upgraded_existing}, "
            f"total {len(merged)} -> {path!r}"
        )
    except Exception as exc:
        return {
            "total": len(merged),
            "added": added,
            "upgraded_existing": upgraded_existing,
            "skipped_existing": skipped,
            "path": path,
            "error": str(exc),
        }

    return {
        "total": len(merged),
        "added": added,
        "upgraded_existing": upgraded_existing,
        "skipped_existing": skipped,
        "path": path,
    }


def normalize_newsapi_content_store(path: str) -> Dict[str, Any]:
    """
    One-shot formatter for an existing `{pair}_newsapi_content.json` file:
    - Ensures `content` is stored as a single line (whitespace-collapsed).
    - Removes `content_truncated` and `content_extracted_from_url` fields.

    This does NOT fetch the network; it only rewrites formatting.
    """
    raw = _load_json_file_best_effort(path)
    articles: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        articles = [a for a in raw if isinstance(a, dict)]
    elif isinstance(raw, dict) and isinstance(raw.get("articles"), list):
        articles = [a for a in raw["articles"] if isinstance(a, dict)]

    changed = 0
    for a in articles:
        before = a.get("content")
        a["content"] = _normalize_single_line(a.get("content") or "")
        a.pop("content_truncated", None)
        a.pop("content_extracted_from_url", None)
        if before != a.get("content"):
            changed += 1

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Normalized NewsAPI content store: changed {changed}, total {len(articles)} -> {path!r}")
        return {"total": len(articles), "changed": changed, "path": path}
    except Exception as exc:
        return {"total": len(articles), "changed": changed, "path": path, "error": str(exc)}


def update_alphavantage_content_store(asset: str, path: str, items: List["NewsItem"]) -> Dict[str, Any]:
    """
    Append-only content store for Alpha Vantage items, de-duplicated by URL.
    Stores:
    - source, title, url, publishedAt, content (single line)
    For new URLs, tries to fetch full text; falls back to the snippet.
    """
    existing_raw = _load_json_file_best_effort(path)
    existing_articles: List[Dict[str, Any]] = []
    if isinstance(existing_raw, list):
        existing_articles = [a for a in existing_raw if isinstance(a, dict)]
    elif isinstance(existing_raw, dict) and isinstance(existing_raw.get("articles"), list):
        existing_articles = [a for a in existing_raw["articles"] if isinstance(a, dict)]

    existing_urls = {
        a.get("url") for a in existing_articles if isinstance(a.get("url"), str) and a.get("url")
    }

    added = 0
    skipped = 0
    new_articles: List[Dict[str, Any]] = []

    for it in items:
        if not isinstance(it.url, str) or not it.url:
            continue
        if it.url in existing_urls:
            skipped += 1
            continue

        body = fetch_article_full_text(it.url)
        if body:
            content = _normalize_single_line(body)
        else:
            content = _normalize_single_line(it.snippet or "")

        new_articles.append(
            {
                "source": it.source,
                "title": it.title,
                "url": it.url,
                "publishedAt": it.published_at,
                "content": content,
            }
        )
        existing_urls.add(it.url)
        added += 1

    merged = existing_articles + new_articles
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Updated Alpha Vantage content store: added {added}, total {len(merged)} -> {path!r}")
    except Exception as exc:
        return {"total": len(merged), "added": added, "skipped_existing": skipped, "path": path, "error": str(exc)}

    return {"total": len(merged), "added": added, "skipped_existing": skipped, "path": path}


def update_eodhd_content_store(asset: str, path: str, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Append-only content store for EODHD news articles, de-duplicated by link.
    Stores: source(host), title, url, publishedAt, content (single line).
    """
    existing_raw = _load_json_file_best_effort(path)
    existing_articles: List[Dict[str, Any]] = []
    if isinstance(existing_raw, list):
        existing_articles = [a for a in existing_raw if isinstance(a, dict)]
    elif isinstance(existing_raw, dict) and isinstance(existing_raw.get("articles"), list):
        existing_articles = [a for a in existing_raw["articles"] if isinstance(a, dict)]

    existing_urls = {
        a.get("url") for a in existing_articles if isinstance(a.get("url"), str) and a.get("url")
    }

    added = 0
    skipped = 0
    new_articles: List[Dict[str, Any]] = []

    for a in articles:
        if not isinstance(a, dict):
            continue
        url = (a.get("link") or "").strip() if isinstance(a.get("link"), str) else ""
        if not url:
            continue
        if url in existing_urls:
            skipped += 1
            continue

        title = (a.get("title") or "").strip()
        published_at = a.get("date")
        content = _normalize_single_line(a.get("content") or "")
        host = _hostname_from_url(url)
        source = host or "eodhd"

        new_articles.append(
            {
                "source": source,
                "title": title,
                "url": url,
                "publishedAt": published_at,
                "content": content,
            }
        )
        existing_urls.add(url)
        added += 1

    merged = existing_articles + new_articles
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Updated EODHD content store: added {added}, total {len(merged)} -> {path!r}")
    except Exception as exc:
        return {"total": len(merged), "added": added, "skipped_existing": skipped, "path": path, "error": str(exc)}

    return {"total": len(merged), "added": added, "skipped_existing": skipped, "path": path}

def _filter_api_source_items_by_urls(
    news: List["NewsItem"],
    *,
    source_prefix: str,
    selected_urls: set[str],
) -> List["NewsItem"]:
    """
    Keep all non-matching sources, but for items whose source starts with source_prefix,
    keep only those whose URL is in selected_urls.
    """
    if not selected_urls:
        return news
    out: List[NewsItem] = []
    for n in news:
        if isinstance(n.source, str) and n.source.startswith(source_prefix):
            if isinstance(n.url, str) and n.url in selected_urls:
                out.append(n)
        else:
            out.append(n)
    return out

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
    investing_max_items: Optional[int] = None,
    enable_marketwatch: bool = True,
    marketwatch_max_items: Optional[int] = None,
    enable_yahoo: bool = True,
    yahoo_max_items: Optional[int] = None,
    enable_newsapi: bool = False,
    newsapi_max_items: Optional[int] = None,
    enable_alphavantage: bool = False,
    alphavantage_max_items: Optional[int] = None,
    enable_eodhd: bool = False,
    eodhd_max_items: Optional[int] = None,
) -> List[NewsItem]:
    """
    Aggregate news from all configured sources, de-duplicate, and convert
    to NewsItem objects.
    """
    pair = normalize_fx_pair(pair)
    raw_items: List[Dict] = []

    # Currently Investing.com and MarketWatch are used as sources.
    cfg = SUPPORTED_PAIRS.get(pair)
    if not cfg:
        raise SystemExit(
            f"Unsupported currency_pair {pair!r}. "
            f"Supported: {', '.join(sorted(SUPPORTED_PAIRS.keys()))}"
        )

    remaining = max_items
    def effective_cap(remaining_budget: Optional[int], per_source_cap: Optional[int]) -> Optional[int]:
        if isinstance(per_source_cap, int) and per_source_cap > 0:
            return per_source_cap if remaining_budget is None else min(remaining_budget, per_source_cap)
        return remaining_budget

    # 1) Investing.com
    if enable_investing and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, investing_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_investing_pair(
                pair=pair,
                url=cfg["investing_url"],
                start_date=start_date,
                end_date=end_date,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 2) MarketWatch (only if we still want more items and URL is configured)
    mw_url = cfg.get("marketwatch_url")
    if enable_marketwatch and mw_url and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, marketwatch_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_marketwatch_pair(
                pair=pair,
                url=mw_url,
                start_date=start_date,
                end_date=end_date,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 3) Yahoo Finance (currently configured only for some pairs, e.g. EUR/USD)
    yahoo_url = cfg.get("yahoo_url")
    if enable_yahoo and yahoo_url and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, yahoo_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_yahoo_pair(
                pair=pair,
                url=yahoo_url,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

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

    # 5) Alpha Vantage NEWS_SENTIMENT (optional API source)
    if enable_alphavantage and (remaining is None or remaining > 0):
        cap = (
            alphavantage_max_items
            if isinstance(alphavantage_max_items, int) and alphavantage_max_items > 0
            else 10
        )
        before = len(raw_items)
        av_items = fetch_alphavantage_fx_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_items=cap,
        )
        if remaining is None:
            raw_items.extend(av_items)
        else:
            raw_items.extend(av_items[:remaining])
        added = len(raw_items) - before
        if remaining is not None:
            remaining = max(0, remaining - added)

    # 6) EODHD News API (optional API source)
    if enable_eodhd and (remaining is None or remaining > 0):
        cap = eodhd_max_items if isinstance(eodhd_max_items, int) and eodhd_max_items > 0 else 10
        before = len(raw_items)
        eodhd_symbol = _eodhd_symbol_for_fx_pair(pair)
        eodhd_items = fetch_eodhd_news(
            asset_label=pair,
            eodhd_symbol=eodhd_symbol,
            start_date=start_date,
            end_date=end_date,
            limit=cap,
            offset=0,
        )
        if remaining is None:
            raw_items.extend(eodhd_items)
        else:
            raw_items.extend(eodhd_items[:remaining])
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
    investing_max_items: Optional[int] = None,
    enable_marketwatch: bool = True,
    marketwatch_max_items: Optional[int] = None,
    enable_yahoo: bool = True,
    yahoo_max_items: Optional[int] = None,
    enable_alphavantage: bool = False,
    alphavantage_max_items: Optional[int] = None,
    enable_eodhd: bool = False,
    eodhd_max_items: Optional[int] = None,
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
    def effective_cap(remaining_budget: Optional[int], per_source_cap: Optional[int]) -> Optional[int]:
        if isinstance(per_source_cap, int) and per_source_cap > 0:
            return per_source_cap if remaining_budget is None else min(remaining_budget, per_source_cap)
        return remaining_budget

    # 1) Investing.com
    investing_url = cfg.get("investing_url")
    if enable_investing and investing_url and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, investing_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_investing_stock(
                symbol=symbol,
                url=investing_url,
                start_date=start_date,
                end_date=end_date,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 2) MarketWatch
    mw_url = cfg.get("marketwatch_url")
    if enable_marketwatch and mw_url and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, marketwatch_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_marketwatch_stock(
                symbol=symbol,
                url=mw_url,
                start_date=start_date,
                end_date=end_date,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 3) Yahoo Finance
    yahoo_url = cfg.get("yahoo_url")
    if enable_yahoo and yahoo_url and (remaining is None or remaining > 0):
        cap = effective_cap(remaining, yahoo_max_items)
        before = len(raw_items)
        raw_items.extend(
            scrape_yahoo_stock(
                symbol=symbol,
                url=yahoo_url,
                start_date=start_date,
                end_date=end_date,
                max_items=cap,
            )
        )
        if remaining is not None:
            added = len(raw_items) - before
            remaining = max(0, remaining - added)

    # 4) Alpha Vantage NEWS_SENTIMENT (optional API source)
    if enable_alphavantage and (remaining is None or remaining > 0):
        cap = (
            alphavantage_max_items
            if isinstance(alphavantage_max_items, int) and alphavantage_max_items > 0
            else 10
        )
        before = len(raw_items)
        av_items = fetch_alphavantage_stock(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            max_items=cap,
        )
        if remaining is None:
            raw_items.extend(av_items)
        else:
            raw_items.extend(av_items[:remaining])
        added = len(raw_items) - before
        if remaining is not None:
            remaining = max(0, remaining - added)

    # 5) EODHD News API (optional API source)
    if enable_eodhd and (remaining is None or remaining > 0):
        cap = eodhd_max_items if isinstance(eodhd_max_items, int) and eodhd_max_items > 0 else 10
        before = len(raw_items)
        eodhd_symbol = _eodhd_symbol_for_stock(symbol)
        eodhd_items = fetch_eodhd_news(
            asset_label=symbol,
            eodhd_symbol=eodhd_symbol,
            start_date=start_date,
            end_date=end_date,
            limit=cap,
            offset=0,
        )
        for it in eodhd_items:
            it["asset_type"] = "stock"
            it["currency_pair"] = symbol
        if remaining is None:
            raw_items.extend(eodhd_items)
        else:
            raw_items.extend(eodhd_items[:remaining])
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
    pre_analysis: Optional[Dict] = None,
) -> None:
    """
    Save crawled news to JSON.

    Backward compatible:
    - If meta/analysis are not provided, writes a simple list (legacy format).
    - If meta and/or analysis are provided, writes a report object:
      {schema_version, meta, news, analysis, pre_analysis}
    """
    payload_news = [asdict(item) for item in news]
    if meta is None and analysis is None and pre_analysis is None:
        payload = payload_news
    else:
        payload = {
            "schema_version": 2,
            "meta": meta or {},
            "news": payload_news,
            "analysis": analysis or {},
            "pre_analysis": pre_analysis or {},
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


def load_default_config() -> Dict[str, object]:
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
            "investing_max_items": None,
            "enable_marketwatch": True,
            "marketwatch_max_items": None,
            "enable_yahoo": True,
            "yahoo_max_items": None,
            "enable_newsapi": False,
            "newsapi_max_items": None,
            "enable_alphavantage": False,
            "alphavantage_max_items": 10,
            "enable_eodhd": False,
            "eodhd_max_items": 10,
            "enable_pre_analysis": True,
            "pre_analysis_max_articles": 20,
            "pre_analysis_filter_for_report": True,
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
            "investing_max_items": None,
            "enable_marketwatch": True,
            "marketwatch_max_items": None,
            "enable_yahoo": True,
            "yahoo_max_items": None,
            "enable_newsapi": False,
            "newsapi_max_items": None,
            "enable_alphavantage": False,
            "alphavantage_max_items": 10,
            "enable_eodhd": False,
            "eodhd_max_items": 10,
            "enable_pre_analysis": True,
            "pre_analysis_max_articles": 20,
            "pre_analysis_filter_for_report": True,
            "enabled_llm_providers": ["deepseek", "openai"],
        }

    start_date = data.get("start_date")
    end_date = data.get("end_date")
    asset_type = data.get("asset_type", "currency")
    if asset_type not in ("currency", "stock"):
        asset_type = "currency"
    
    currency_pair = normalize_fx_pair(data.get("currency_pair", "EUR/USD"))
    stock = data.get("stock")
    max_news = data.get("max_news", None)
    enable_investing = bool(data.get("enable_investing", True))
    investing_max_items = data.get("investing_max_items", None)
    if not (isinstance(investing_max_items, int) and investing_max_items > 0):
        investing_max_items = None
    enable_marketwatch = bool(data.get("enable_marketwatch", True))
    marketwatch_max_items = data.get("marketwatch_max_items", None)
    if not (isinstance(marketwatch_max_items, int) and marketwatch_max_items > 0):
        marketwatch_max_items = None
    enable_yahoo = bool(data.get("enable_yahoo", True))
    yahoo_max_items = data.get("yahoo_max_items", None)
    if not (isinstance(yahoo_max_items, int) and yahoo_max_items > 0):
        yahoo_max_items = None
    enable_newsapi = bool(data.get("enable_newsapi", False))
    enable_alphavantage = bool(data.get("enable_alphavantage", False))
    enable_eodhd = bool(data.get("enable_eodhd", False))
    enable_pre_analysis = bool(data.get("enable_pre_analysis", True))
    pre_analysis_max_articles = data.get("pre_analysis_max_articles", 20)
    pre_analysis_filter_for_report = bool(data.get("pre_analysis_filter_for_report", True))
    newsapi_max_items = data.get("newsapi_max_items", None)
    if not (isinstance(newsapi_max_items, int) and newsapi_max_items > 0):
        newsapi_max_items = None
    alphavantage_max_items = data.get("alphavantage_max_items", 10)
    if not (isinstance(alphavantage_max_items, int) and alphavantage_max_items > 0):
        alphavantage_max_items = 10

    eodhd_max_items = data.get("eodhd_max_items", 10)
    if not (isinstance(eodhd_max_items, int) and eodhd_max_items > 0):
        eodhd_max_items = 10
    
    if asset_type == "currency":
        if currency_pair not in SUPPORTED_PAIRS:
            currency_pair = "EUR/USD"
    elif asset_type == "stock":
        if not stock or stock not in SUPPORTED_STOCKS:
            stock = "AAPL"  # Default to Apple if invalid stock

    if not (isinstance(pre_analysis_max_articles, int) and pre_analysis_max_articles > 0):
        pre_analysis_max_articles = 20

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
        "investing_max_items": investing_max_items,
        "enable_marketwatch": enable_marketwatch,
        "marketwatch_max_items": marketwatch_max_items,
        "enable_yahoo": enable_yahoo,
        "yahoo_max_items": yahoo_max_items,
        "enable_newsapi": enable_newsapi,
        "newsapi_max_items": newsapi_max_items,
        "enable_alphavantage": enable_alphavantage,
        "alphavantage_max_items": alphavantage_max_items,
        "enable_eodhd": enable_eodhd,
        "eodhd_max_items": eodhd_max_items,
        "enable_pre_analysis": enable_pre_analysis,
        "pre_analysis_max_articles": pre_analysis_max_articles,
        "pre_analysis_filter_for_report": pre_analysis_filter_for_report,
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


def _try_parse_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Best-effort JSON object parser for LLM responses.

    Returns: (parsed_dict_or_None, error_message_or_None)
    """
    if not text:
        return None, "empty response"

    cleaned = text.strip()

    # Strip common Markdown code fences.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

    # Fast path: full JSON
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj, None
        return None, "top-level JSON is not an object"
    except Exception:
        pass

    # Fallback: extract the first {...} block.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no JSON object found"
    candidate = cleaned[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj, None
        return None, "extracted JSON is not an object"
    except Exception as exc:
        return None, f"failed to parse JSON: {exc}"


def build_pre_analysis_prompt(news_context: str, pair: str) -> str:
    return (
        "You are a financial news quality rater for FX trading.\n"
        f"Your task: compare and judge the quality of the collected news for {pair} "
        "across different sources.\n\n"
        "Return STRICT JSON (no markdown, no code fences, no extra text).\n"
        "The JSON must follow this schema exactly:\n"
        "{\n"
        '  "pair": string,\n'
        '  "source_summary": { "<source>": {"coverage": 0-10, "credibility": 0-10, "signal": 0-10, "notes": string} },\n'
        '  "themes": [ {"theme": string, "importance": 0-10, "evidence_news_ids": [int]} ],\n'
        '  "duplicates": [ {"group_id": string, "news_ids": [int], "canonical_news_id": int, "notes": string} ],\n'
        '  "article_quality": [\n'
        "    {\n"
        '      "news_id": int,\n'
        '      "source": string,\n'
        '      "relevance": 0-10,\n'
        '      "credibility": 0-10,\n'
        '      "timeliness": 0-10,\n'
        '      "signal": 0-10,\n'
        '      "overall": 0-10,\n'
        '      "is_duplicate": boolean,\n'
        '      "duplicate_group_id": string | null,\n'
        '      "reasons": [string]\n'
        "    }\n"
        "  ],\n"
        '  "recommended_news_ids": [int],\n'
        '  "excluded": [ {"news_id": int, "reason": string} ],\n'
        '  "cross_source_comparison": {\n'
        '     "agreements": [string],\n'
        '     "conflicts": [string],\n'
        '     "missing_coverage": [string],\n'
        '     "best_sources": [string]\n'
        "  }\n"
        "}\n\n"
        "Scoring guidance:\n"
        "- credibility: original reporting / named sources / data-driven > opinion-only\n"
        "- signal: macro, central bank, inflation, growth, risk sentiment, positioning > generic market recap\n"
        "- timeliness: within the requested date window and latest updates score higher\n"
        "- duplicates: group near-identical stories across sources\n"
        "- recommended_news_ids: keep the smallest set that covers all key themes with high overall score\n\n"
        "Here is the news context:\n\n"
        f"{news_context}\n"
    )


def run_deepseek_pre_analysis(
    news: List[NewsItem],
    *,
    pair: str,
    max_articles: int = 20,
) -> Dict[str, Any]:
    """
    DeepSeek-only pre-analysis: evaluate news quality + cross-source comparison.
    Returns a dict; on failure, returns {"error": "..."}.
    """
    if not news:
        return {"error": "No news provided for pre-analysis."}

    client = make_deepseek_client()
    if client is None:
        return {"error": "DEEPSEEK_API_KEY not set; pre-analysis skipped."}

    context = build_news_context_for_llm(
        news, max_articles=max_articles, fetch_full_paragraphs=True
    )
    prompt = build_pre_analysis_prompt(context, pair)

    print("\n[INFO] Requesting DeepSeek pre-analysis (quality + cross-source comparison)...")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or ""
        parsed, err = _try_parse_json_object(raw)
        if parsed is None:
            return {"error": f"Failed to parse pre-analysis JSON: {err}", "raw": raw}
        return parsed
    except Exception as exc:
        return {"error": f"Error calling DeepSeek for pre-analysis: {exc}"}


def build_source_relevance_prompt(asset: str, source_name: str, articles: List[Dict[str, Any]]) -> str:
    """
    Ask DeepSeek to classify source articles by relevance to the asset (FX pair),
    and to rate their quality.
    """
    blocks: List[str] = []
    for idx, a in enumerate(articles, start=1):
        if not isinstance(a, dict):
            continue
        url = a.get("url") or ""
        title = a.get("title") or ""
        published_at = a.get("publishedAt") or ""
        source = a.get("source")
        if isinstance(source, dict):
            source = source.get("name")
        content = a.get("content") or ""
        if isinstance(content, str) and len(content) > 1200:
            content = content[:1200] + "..."

        blocks.append(
            f"Article {idx}:\n"
            f"URL: {url}\n"
            f"Source: {source}\n"
            f"Time: {published_at}\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
            f"---"
        )

    context = "\n\n".join(blocks)

    return (
        "You are a financial news relevance judge for FX trading.\n"
        f"Classify {source_name} articles by how relevant they are to {asset}, and rate their quality.\n\n"
        "Relevance rules:\n"
        "- most_relevant: directly about the pair OR clearly about BOTH currencies and their exchange rate drivers\n"
        "- relevant: clearly about ONLY ONE of the two currencies (EUR-only or USD-only), or a major USD/EUR driver but not explicitly the pair\n"
        "- unrelated: not directly relevant to EUR or USD or FX; generic equity/crypto/corporate news without FX linkage\n\n"
        "Return STRICT JSON only (no markdown, no extra text):\n"
        "{\n"
        '  "pair": string,\n'
        '  "counts": {"most_relevant": int, "relevant": int, "unrelated": int},\n'
        '  "selected_urls": [string],\n'
        '  "items": [\n'
        "    {\n"
        '      "url": string,\n'
        '      "label": "most_relevant" | "relevant" | "unrelated",\n'
        '      "relevance_score": 0-10,\n'
        '      "quality_score": 0-10,\n'
        '      "reasons": [string]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Important:\n"
        "- Only include URLs that appear in the context.\n"
        "- selected_urls should include ONLY most_relevant + relevant (exclude unrelated).\n"
        "- Order selected_urls from best to worst using: (label, relevance_score, quality_score).\n\n"
        f"Here is the {source_name} context:\n\n"
        f"{context}\n"
    )


def build_newsapi_relevance_prompt(pair: str, articles: List[Dict[str, Any]]) -> str:
    """
    Ask DeepSeek to classify NewsAPI articles by relevance to the FX pair.
    """
    return build_source_relevance_prompt(pair, "NewsAPI", articles)


def run_deepseek_newsapi_relevance(
    *,
    pair: str,
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    DeepSeek-only: classify NewsAPI articles by relevance to the FX pair.
    """
    client = make_deepseek_client()
    if client is None:
        return {"error": "DEEPSEEK_API_KEY not set; NewsAPI relevance analysis skipped."}
    if not articles:
        return {"error": "No NewsAPI articles provided for relevance analysis."}

    prompt = build_newsapi_relevance_prompt(pair, articles)
    print("\n[INFO] Requesting DeepSeek NewsAPI relevance analysis...")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1800,
        )
        raw = resp.choices[0].message.content or ""
        parsed, err = _try_parse_json_object(raw)
        if parsed is None:
            return {"error": f"Failed to parse NewsAPI relevance JSON: {err}", "raw": raw}
        return parsed
    except Exception as exc:
        return {"error": f"Error calling DeepSeek for NewsAPI relevance: {exc}"}


def run_deepseek_source_relevance(
    *,
    asset: str,
    source_name: str,
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    client = make_deepseek_client()
    if client is None:
        return {"error": "DEEPSEEK_API_KEY not set; relevance analysis skipped."}
    if not articles:
        return {"error": f"No {source_name} articles provided for relevance analysis."}

    prompt = build_source_relevance_prompt(asset, source_name, articles)
    print(f"\n[INFO] Requesting DeepSeek {source_name} relevance analysis...")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1800,
        )
        raw = resp.choices[0].message.content or ""
        parsed, err = _try_parse_json_object(raw)
        if parsed is None:
            return {"error": f"Failed to parse {source_name} relevance JSON: {err}", "raw": raw}
        return parsed
    except Exception as exc:
        return {"error": f"Error calling DeepSeek for {source_name} relevance: {exc}"}


def save_newsapi_relevance_json(pair: str, path: str, payload: Dict[str, Any]) -> bool:
    try:
        out = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "pair": pair,
            "relevance": payload,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved NewsAPI relevance analysis to {path!r}")
        return True
    except Exception as exc:
        debug_log(f"[WARN] Failed to save NewsAPI relevance JSON: {exc}")
        return False


def save_source_relevance_json(asset: str, path: str, *, source_name: str, payload: Dict[str, Any]) -> bool:
    try:
        out = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "source": source_name,
            "relevance": payload,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {source_name} relevance analysis to {path!r}")
        return True
    except Exception as exc:
        debug_log(f"[WARN] Failed to save {source_name} relevance JSON: {exc}")
        return False


def save_individual_llm_outputs(
    *,
    asset: str,
    asset_type: str,
    results: Dict[str, str],
    output_prefix: str,
) -> None:
    """
    Save each provider's analysis into its own JSON file, e.g.:
    - eur_usd_deepseek.json
    - eur_usd_chatgpt.json
    - eur_usd_gemini.json
    """
    provider_to_filename = {
        "deepseek": "deepseek",
        "openai": "chatgpt",
        "gemini": "gemini",
    }

    for provider, text in results.items():
        if not isinstance(text, str):
            continue
        suffix = provider_to_filename.get(provider, provider)
        path = f"{output_prefix}_{suffix}.json"
        try:
            payload = {
                "schema_version": 1,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "asset_type": asset_type,
                "asset": asset,
                "provider": provider,
                "output": text,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Saved {provider} analysis output to {path!r}")
        except Exception as exc:
            debug_log(f"[WARN] Failed saving {provider} output JSON: {exc}")

def _format_pre_analysis_summary(pre: Dict[str, Any]) -> str:
    """
    Compact summary injected into downstream LLM prompts.
    """
    if not pre or "error" in pre:
        return ""

    pair = pre.get("pair", "")
    best_sources = []
    csc = pre.get("cross_source_comparison")
    if isinstance(csc, dict) and isinstance(csc.get("best_sources"), list):
        best_sources = [str(x) for x in csc["best_sources"] if isinstance(x, str)]

    themes = pre.get("themes")
    top_themes: List[str] = []
    if isinstance(themes, list):
        for t in themes[:5]:
            if isinstance(t, dict) and isinstance(t.get("theme"), str):
                top_themes.append(t["theme"])

    recommended = pre.get("recommended_news_ids")
    rec_count = len(recommended) if isinstance(recommended, list) else 0

    parts = [
        "DeepSeek pre-analysis (news quality + cross-source comparison):",
        f"- Pair: {pair}" if pair else "- Pair: (unknown)",
        f"- Best sources: {', '.join(best_sources)}" if best_sources else "- Best sources: (not specified)",
        f"- Key themes: {', '.join(top_themes)}" if top_themes else "- Key themes: (not specified)",
        f"- Recommended articles: {rec_count}",
        "Use this to prioritize high-signal, credible, non-duplicate items.",
    ]
    return "\n".join(parts) + "\n\n"


def _select_news_by_ids(news: List[NewsItem], ids: List[int]) -> List[NewsItem]:
    id_to_item = {n.id: n for n in news}
    selected: List[NewsItem] = []
    for i in ids:
        if isinstance(i, int) and i in id_to_item:
            selected.append(id_to_item[i])
    return selected


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


def run_llm_analysis(
    news: List[NewsItem],
    asset: str,
    asset_type: str = "currency",
    enabled_providers: Optional[List[str]] = None,
    max_articles: int = 8,
    *,
    pre_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    if not news:
        return {"error": "No news available for analysis."}
    if enabled_providers is None:
        enabled_providers = ["deepseek", "openai"]

    context = build_news_context_for_llm(
        news, max_articles=max_articles, fetch_full_paragraphs=True
    )
    pre_summary = _format_pre_analysis_summary(pre_analysis or {})
    prompt = pre_summary + build_analysis_prompt(context, asset, asset_type)

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
        "--normalize-newsapi-content-store",
        action="store_true",
        help="Rewrite `{pair}_newsapi_content.json` to make `content` single-line "
             "and remove non-content metadata (no network).",
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
    asset_type = str(cfg.get("asset_type", "currency"))
    if asset_type == "stock":
        stock = str(cfg.get("stock") or "AAPL")
        if stock not in SUPPORTED_STOCKS:
            stock = "AAPL"
        print(f"[INFO] Selected stock from config: {stock}")
        asset = stock
    else:
        pair = str(cfg.get("currency_pair") or "EUR/USD")
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

    # One-shot formatter for an existing NewsAPI content store.
    # Must run BEFORE any crawling (no network).
    if args.normalize_newsapi_content_store:
        if asset_type != "currency":
            print("[INFO] --normalize-newsapi-content-store only applies to FX pairs.")
            return
        safe_asset = asset.replace("/", "_").replace(" ", "").lower()
        if args.output:
            output_path = args.output
        else:
            output_path = f"{safe_asset}_news.json"

        base_no_ext, _ = os.path.splitext(output_path)
        if base_no_ext.endswith("_news"):
            newsapi_content_path = base_no_ext[:-5] + "_newsapi_content.json"
        else:
            newsapi_content_path = base_no_ext + "_newsapi_content.json"
        normalize_newsapi_content_store(newsapi_content_path)
        return

    # Collect news based on asset type
    if asset_type == "stock":
        news_items = collect_news_for_stock(
            asset,
            start_date,
            end_date,
            max_items=max_news,
            enable_investing=bool(cfg.get("enable_investing", True)),
            investing_max_items=cfg.get("investing_max_items", None),
            enable_marketwatch=bool(cfg.get("enable_marketwatch", True)),
            marketwatch_max_items=cfg.get("marketwatch_max_items", None),
            enable_yahoo=bool(cfg.get("enable_yahoo", True)),
            yahoo_max_items=cfg.get("yahoo_max_items", None),
            enable_alphavantage=bool(cfg.get("enable_alphavantage", False)),
            alphavantage_max_items=cfg.get("alphavantage_max_items", 10),
            enable_eodhd=bool(cfg.get("enable_eodhd", False)),
            eodhd_max_items=cfg.get("eodhd_max_items", 10),
        )
    else:
        news_items = collect_news_for_pair(
            asset,
            start_date,
            end_date,
            max_items=max_news,
            enable_investing=bool(cfg.get("enable_investing", True)),
            investing_max_items=cfg.get("investing_max_items", None),
            enable_marketwatch=bool(cfg.get("enable_marketwatch", True)),
            marketwatch_max_items=cfg.get("marketwatch_max_items", None),
            enable_yahoo=bool(cfg.get("enable_yahoo", True)),
            yahoo_max_items=cfg.get("yahoo_max_items", None),
            enable_newsapi=bool(cfg.get("enable_newsapi", False)),
            newsapi_max_items=cfg.get("newsapi_max_items", None),
            enable_alphavantage=bool(cfg.get("enable_alphavantage", False)),
            alphavantage_max_items=cfg.get("alphavantage_max_items", 10),
            enable_eodhd=bool(cfg.get("enable_eodhd", False)),
            eodhd_max_items=cfg.get("eodhd_max_items", 10),
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

    meta: Dict[str, Any] = {
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
            "alphavantage": bool(cfg.get("enable_alphavantage", False)),
            "eodhd": bool(cfg.get("enable_eodhd", False)),
        },
        "source_max_items": {
            "investing": cfg.get("investing_max_items", None),
            "marketwatch": cfg.get("marketwatch_max_items", None),
            "yahoo": cfg.get("yahoo_max_items", None),
            "newsapi": cfg.get("newsapi_max_items", None),
            "alphavantage": cfg.get("alphavantage_max_items", 10),
            "eodhd": cfg.get("eodhd_max_items", 10),
        },
        "newsapi_max_items": cfg.get("newsapi_max_items", None),
        "llm_enabled": cfg.get("enabled_llm_providers", []),
        "pre_analysis": {
            "enabled": bool(cfg.get("enable_pre_analysis", True)),
            "max_articles": cfg.get("pre_analysis_max_articles", 20),
            "filter_for_report": bool(cfg.get("pre_analysis_filter_for_report", True)),
        },
    }

    pre_max_articles = meta["pre_analysis"].get("max_articles")
    if not (isinstance(pre_max_articles, int) and pre_max_articles > 0):
        pre_max_articles = 20
        meta["pre_analysis"]["max_articles"] = pre_max_articles

    newsapi_relevance: Optional[Dict[str, Any]] = None
    newsapi_selected_urls: set[str] = set()
    alphavantage_relevance: Optional[Dict[str, Any]] = None
    alphavantage_selected_urls: set[str] = set()

    # Additionally save a separate NewsAPI-only JSON with full content for future access/reuse.
    if asset_type == "currency" and bool(cfg.get("enable_newsapi", False)):
        base_no_ext, _ = os.path.splitext(output_path)
        if base_no_ext.endswith("_news"):
            newsapi_content_path = base_no_ext[:-5] + "_newsapi_content.json"
            newsapi_relevance_path = base_no_ext[:-5] + "_newsapi_quality.json"
        else:
            newsapi_content_path = base_no_ext + "_newsapi_content.json"
            newsapi_relevance_path = base_no_ext + "_newsapi_quality.json"

        # 1) Append-only content store
        update_newsapi_content_store(asset, newsapi_content_path)

        # 2) DeepSeek relevance/quality classification (for THIS run's NewsAPI articles)
        cached = _NEWSAPI_FULL_CACHE.get(asset)
        cached_articles = cached.get("articles") if isinstance(cached, dict) else None
        if isinstance(cached_articles, list) and cached_articles:
            cap = cfg.get("newsapi_max_items", None)
            max_for_relevance = cap if isinstance(cap, int) and cap > 0 else 12
            newsapi_relevance = run_deepseek_newsapi_relevance(
                pair=asset,
                articles=cached_articles[:max_for_relevance],
            )
            save_newsapi_relevance_json(asset, newsapi_relevance_path, newsapi_relevance)

            counts = newsapi_relevance.get("counts") if isinstance(newsapi_relevance, dict) else None
            if isinstance(counts, dict):
                mr = counts.get("most_relevant", 0)
                r = counts.get("relevant", 0)
                u = counts.get("unrelated", 0)
                print(f"[INFO] NewsAPI relevance counts (once): most_relevant={mr}, relevant={r}, unrelated={u}")

            sel = newsapi_relevance.get("selected_urls") if isinstance(newsapi_relevance, dict) else None
            if isinstance(sel, list):
                newsapi_selected_urls = {u for u in sel if isinstance(u, str) and u}

    # Alpha Vantage: save content store + DeepSeek relevance/quality (similar to NewsAPI)
    if bool(cfg.get("enable_alphavantage", False)):
        base_no_ext, _ = os.path.splitext(output_path)
        if base_no_ext.endswith("_news"):
            av_content_path = base_no_ext[:-5] + "_alphavantage_content.json"
            av_quality_path = base_no_ext[:-5] + "_alphavantage_quality.json"
        else:
            av_content_path = base_no_ext + "_alphavantage_content.json"
            av_quality_path = base_no_ext + "_alphavantage_quality.json"

        av_items = [n for n in news_items if isinstance(n.source, str) and n.source.startswith("AlphaVantage")]
        if av_items:
            update_alphavantage_content_store(asset, av_content_path, av_items)

            # Build per-run article dicts for DeepSeek relevance/quality scoring.
            av_articles: List[Dict[str, Any]] = []
            for n in av_items:
                av_articles.append(
                    {
                        "url": n.url,
                        "source": n.source,
                        "publishedAt": n.published_at,
                        "title": n.title,
                        "content": n.snippet or "",
                    }
                )
            max_for_relevance = int(cfg.get("alphavantage_max_items", 10) or 10)
            if max_for_relevance <= 0:
                max_for_relevance = 10
            alphavantage_relevance = run_deepseek_source_relevance(
                asset=asset,
                source_name="AlphaVantage",
                articles=av_articles[:max_for_relevance],
            )
            save_source_relevance_json(asset, av_quality_path, source_name="AlphaVantage", payload=alphavantage_relevance)

            counts = alphavantage_relevance.get("counts") if isinstance(alphavantage_relevance, dict) else None
            if isinstance(counts, dict):
                mr = counts.get("most_relevant", 0)
                r = counts.get("relevant", 0)
                u = counts.get("unrelated", 0)
                print(f"[INFO] AlphaVantage relevance counts (once): most_relevant={mr}, relevant={r}, unrelated={u}")

            sel = alphavantage_relevance.get("selected_urls") if isinstance(alphavantage_relevance, dict) else None
            if isinstance(sel, list):
                alphavantage_selected_urls = {u for u in sel if isinstance(u, str) and u}

    # EODHD: save content store + DeepSeek relevance/quality (similar to NewsAPI)
    eodhd_selected_urls: set[str] = set()
    if bool(cfg.get("enable_eodhd", False)):
        base_no_ext, _ = os.path.splitext(output_path)
        if base_no_ext.endswith("_news"):
            eodhd_content_path = base_no_ext[:-5] + "_eodhd_content.json"
            eodhd_quality_path = base_no_ext[:-5] + "_eodhd_quality.json"
        else:
            eodhd_content_path = base_no_ext + "_eodhd_content.json"
            eodhd_quality_path = base_no_ext + "_eodhd_quality.json"

        cached = _EODHD_FULL_CACHE.get(asset)
        cached_articles = cached.get("articles") if isinstance(cached, dict) else None
        if isinstance(cached_articles, list) and cached_articles:
            update_eodhd_content_store(asset, eodhd_content_path, cached_articles)

            # Build per-run dicts for DeepSeek.
            eodhd_articles_for_llm: List[Dict[str, Any]] = []
            for a in cached_articles:
                if not isinstance(a, dict):
                    continue
                eodhd_articles_for_llm.append(
                    {
                        "url": a.get("link"),
                        "source": _hostname_from_url(a.get("link") or ""),
                        "publishedAt": a.get("date"),
                        "title": a.get("title"),
                        "content": a.get("content") or "",
                    }
                )
            max_for_relevance = int(cfg.get("eodhd_max_items", 10) or 10)
            if max_for_relevance <= 0:
                max_for_relevance = 10
            eodhd_relevance = run_deepseek_source_relevance(
                asset=asset,
                source_name="EODHD",
                articles=eodhd_articles_for_llm[:max_for_relevance],
            )
            save_source_relevance_json(asset, eodhd_quality_path, source_name="EODHD", payload=eodhd_relevance)

            counts = eodhd_relevance.get("counts") if isinstance(eodhd_relevance, dict) else None
            if isinstance(counts, dict):
                mr = counts.get("most_relevant", 0)
                r = counts.get("relevant", 0)
                u = counts.get("unrelated", 0)
                print(f"[INFO] EODHD relevance counts (once): most_relevant={mr}, relevant={r}, unrelated={u}")

            sel = eodhd_relevance.get("selected_urls") if isinstance(eodhd_relevance, dict) else None
            if isinstance(sel, list):
                eodhd_selected_urls = {u for u in sel if isinstance(u, str) and u}

    if args.no_llm:
        save_news_to_json(news_items, output_path, meta=meta, analysis={}, pre_analysis={})
        print("[INFO] Skipping LLM analysis (--no-llm specified).")
        return

    pre_analysis: Dict[str, Any] = {}
    analysis_news = news_items
    if asset_type == "currency" and bool(cfg.get("enable_pre_analysis", True)):
        pre_analysis = run_deepseek_pre_analysis(
            news_items,
            pair=asset,
            max_articles=pre_max_articles,
        )
        rec_ids = pre_analysis.get("recommended_news_ids") if isinstance(pre_analysis, dict) else None
        if (
            bool(cfg.get("pre_analysis_filter_for_report", True))
            and isinstance(rec_ids, list)
            and rec_ids
        ):
            analysis_news = _select_news_by_ids(news_items, rec_ids)
            if analysis_news:
                meta["analysis_input_news_ids"] = [n.id for n in analysis_news]
                meta["pre_analysis_recommended_news_ids"] = rec_ids

    # If NewsAPI is enabled and we have a relevance classification for this run,
    # filter only the API-source items (keep other sources).
    if asset_type == "currency" and bool(cfg.get("enable_newsapi", False)):
        if newsapi_selected_urls:
            analysis_news = _filter_api_source_items_by_urls(
                analysis_news,
                source_prefix="NewsAPI",
                selected_urls=newsapi_selected_urls,
            )
            meta["newsapi_selected_urls_for_report"] = sorted(newsapi_selected_urls)

    if bool(cfg.get("enable_alphavantage", False)) and alphavantage_selected_urls:
        analysis_news = _filter_api_source_items_by_urls(
            analysis_news,
            source_prefix="AlphaVantage",
            selected_urls=alphavantage_selected_urls,
        )
        meta["alphavantage_selected_urls_for_report"] = sorted(alphavantage_selected_urls)

    if bool(cfg.get("enable_eodhd", False)) and eodhd_selected_urls:
        analysis_news = _filter_api_source_items_by_urls(
            analysis_news,
            source_prefix="EODHD",
            selected_urls=eodhd_selected_urls,
        )
        meta["eodhd_selected_urls_for_report"] = sorted(eodhd_selected_urls)

    results = run_llm_analysis(
        analysis_news,
        asset=asset,
        asset_type=asset_type,
        enabled_providers=cfg.get("enabled_llm_providers", ["deepseek", "openai"]),
        max_articles=args.max_articles,
        pre_analysis=pre_analysis,
    )
    save_news_to_json(
        news_items,
        output_path,
        meta=meta,
        analysis=results,
        pre_analysis=pre_analysis,
    )
    print_analysis_report(results, asset, asset_type)

    # Save each provider output to its own JSON file as well.
    if asset_type == "stock":
        output_prefix = asset.lower()
    else:
        output_prefix = asset.replace("/", "_").replace(" ", "").lower()
    save_individual_llm_outputs(
        asset=asset,
        asset_type=asset_type,
        results=results,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    main()