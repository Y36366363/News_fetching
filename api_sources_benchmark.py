"""
Benchmark the quality of *API* FX news sources:
- NewsAPI.org
- ForexNewsAPI (forexnewsapi.com)
- Alpha Vantage (NEWS_SENTIMENT)
- EODHD News API

This script is intentionally separate from `news_fetching.py`.

What it measures per FX pair & provider (per run):
- Recency lag: how old is the newest article we can retrieve right now?
- Volume/coverage: how many unique related articles are available in the chosen window?
- Quality/reliability (rule-based): a deterministic score (0-10) + reliability tier derived from the publisher domain.
- Optional DeepSeek pass: after rule-based filtering, ask DeepSeek to classify relevance & rate quality.

Notes:
- For historical probing without burning API quotas, you can use `--from-cache`
  to read your append-only content stores: `{pair}_newsapi_content.json`,
  `{pair}_forexnewsapi_content.json`,
  `{pair}_alphavantage_content.json`, `{pair}_eodhd_content.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from dotenv import load_dotenv

# We intentionally reuse your existing, battle-tested fetchers + DeepSeek helpers.
import news_fetching as nf


DEFAULT_FX_PAIRS: List[str] = ["EUR/USD", "USD/JPY", "GBP/USD", "USD/CNY", "USD/CAD", "AUD/USD", "AUD/CAD"]
DEFAULT_LOG_JSONL = os.path.join("benchmarks", "api_benchmark_log.jsonl")


# -----------------------------------------------------------------------------
# Rule-based quality + reliability scoring
# -----------------------------------------------------------------------------

# Very rough "reliability tiers" by domain. You should edit/extend this over time.
# Tiering logic:
# - tier_1: generally strong editorial standards / original reporting
# - tier_2: usually ok but can include republished / mixed quality
# - tier_3: unknown / blog / niche; needs scrutiny
# - low: press-release wires, promo farms, or frequent low-signal domains

TIER_1_HOSTS = {
    "reuters.com",
    "bloomberg.com",
    "ft.com",
    "wsj.com",
    "economist.com",
    "cnbc.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "spglobal.com",
    "imf.org",
    "worldbank.org",
    "ecb.europa.eu",
    "federalreserve.gov",
    "bankofengland.co.uk",
    "boj.or.jp",
    "bis.org",
}

TIER_2_HOSTS = {
    "fxstreet.com",
    "investing.com",
    "tradingview.com",
    "forexlive.com",
    "theedgemalaysia.com",
    "channelnewsasia.com",
    "financialpost.com",
    "bnnbloomberg.ca",
    "barrons.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
}

LOW_TRUST_HOSTS = {
    "globenewswire.com",
    "prnewswire.com",
    "businesswire.com",
    "accesswire.com",
    "newsbtc.com",
    "cointelegraph.com",
    "cryptoslate.com",
    "ambcrypto.com",
    "theflightdeal.com",
}

CLICKBAIT_TITLE_PATTERNS = [
    r"\b(shocking|insane|you won't believe|secret|guaranteed)\b",
    r"\b(to the moon|100x|explodes|skyrockets)\b",
    r"\b(\d{2,}%\s*(gain|rise|profit))\b",
]

FX_DRIVER_TERMS = [
    "forex",
    "fx",
    "currency",
    "exchange rate",
    "central bank",
    "fed",
    "fomc",
    "ecb",
    "boj",
    "boe",
    "pboc",
    "inflation",
    "cpi",
    "ppi",
    "gdp",
    "jobs",
    "payrolls",
    "pmi",
    "yield",
    "treasury",
    "rate cut",
    "rate hike",
    "policy",
]


def _canon_host(host: str) -> str:
    h = (host or "").strip().lower()
    if h.startswith("www."):
        h = h[4:]
    return h


def host_from_url(url: str) -> str:
    try:
        return _canon_host(urlparse(url).netloc or "")
    except Exception:
        return ""


def reliability_tier_for_host(host: str) -> str:
    h = _canon_host(host)
    if not h:
        return "tier_3"
    if h in TIER_1_HOSTS:
        return "tier_1"
    if h in TIER_2_HOSTS:
        return "tier_2"
    if h in LOW_TRUST_HOSTS:
        return "low"
    # Handle common subdomains (e.g. www.reuters.com already handled, but also e.g. uk.reuters.com).
    for base in TIER_1_HOSTS:
        if h.endswith("." + base):
            return "tier_1"
    for base in TIER_2_HOSTS:
        if h.endswith("." + base):
            return "tier_2"
    for base in LOW_TRUST_HOSTS:
        if h.endswith("." + base):
            return "low"
    return "tier_3"


def _pair_tokens(pair: str) -> Tuple[str, str, str]:
    parts = pair.replace(" ", "").upper().split("/")
    base = parts[0] if len(parts) > 0 else pair.upper()
    quote = parts[1] if len(parts) > 1 else ""
    ticker = f"{base}{quote}" if quote else base
    return base, quote, ticker


def _text_contains_any(text: str, terms: Iterable[str]) -> bool:
    t = (text or "").lower()
    return any(term.lower() in t for term in terms)


def _looks_clickbaity(title: str) -> bool:
    t = (title or "").lower()
    return any(re.search(p, t) for p in CLICKBAIT_TITLE_PATTERNS)


@dataclass
class RuleScore:
    score_0_10: int
    reliability_tier: str
    reasons: List[str]


def rule_score_article(*, pair: str, title: str, snippet: str, url: str) -> RuleScore:
    """
    Deterministic, explainable score.
    - 0-10 quality score
    - reliability tier by domain
    """
    base, quote, ticker = _pair_tokens(pair)
    host = host_from_url(url)
    tier = reliability_tier_for_host(host)

    score = 5.0
    reasons: List[str] = []

    # Reliability baseline by domain tier.
    if tier == "tier_1":
        score += 2.5
        reasons.append("tier_1 publisher domain")
    elif tier == "tier_2":
        score += 1.5
        reasons.append("tier_2 publisher domain")
    elif tier == "tier_3":
        score += 0.0
        reasons.append("unknown/uncurated publisher domain (tier_3)")
    else:
        score -= 2.5
        reasons.append("low-trust / promo-heavy domain")

    t = (title or "").strip()
    s = (snippet or "").strip()
    combined = f"{t} {s}".lower()

    # Relevance / signal indicators (still rule-based; DeepSeek can refine later).
    if pair.replace(" ", "").lower() in combined or ticker.lower() in combined:
        score += 1.5
        reasons.append("explicitly mentions the FX pair/ticker")
    else:
        # Mentions both currencies somewhere (weaker than explicit pair).
        if base.lower() in combined and quote.lower() in combined:
            score += 0.8
            reasons.append("mentions both currencies")

    if _text_contains_any(combined, FX_DRIVER_TERMS):
        score += 0.8
        reasons.append("mentions core FX driver terms (rates/macro/CB/etc.)")

    # Content completeness proxies.
    if len(t) >= 20:
        score += 0.4
        reasons.append("non-trivial title length")
    if len(s) >= 120:
        score += 0.6
        reasons.append("has a meaningful snippet/content excerpt")
    elif len(s) == 0:
        score -= 0.6
        reasons.append("missing snippet/content excerpt")

    # Penalties for obvious low-signal / spam patterns.
    if _looks_clickbaity(t):
        score -= 1.0
        reasons.append("clickbait title patterns detected")

    if _text_contains_any(combined, ["press release", "globenewswire", "prnewswire", "business wire"]):
        score -= 1.2
        reasons.append("press release / wire-like content")

    if _text_contains_any(combined, ["airlines", "roundtrip", "hotel", "travel deal"]):
        score -= 2.0
        reasons.append("off-topic lifestyle/travel pattern")

    # Clamp and round.
    score = max(0.0, min(10.0, score))
    return RuleScore(score_0_10=int(round(score)), reliability_tier=tier, reasons=reasons)


# -----------------------------------------------------------------------------
# Cache loading helpers (append-only content stores)
# -----------------------------------------------------------------------------


def _pair_to_prefix(pair: str) -> str:
    return pair.lower().replace("/", "_").replace(" ", "")


def load_cached_provider_articles(
    *,
    pair: str,
    provider: str,
    root_dir: str,
) -> List[Dict[str, Any]]:
    """
    Read append-only cache files produced by `news_fetching.py`:
    - newsapi:  `{pair}_newsapi_content.json` (list of NewsAPI article objects)
    - alphavantage: `{pair}_alphavantage_content.json` (list of normalized objects)
    - forexnewsapi: `{pair}_forexnewsapi_content.json` (list of normalized objects)
    - eodhd: `{pair}_eodhd_content.json` (list of normalized objects)
    """
    prefix = _pair_to_prefix(pair)
    provider = provider.lower().strip()
    if provider == "newsapi":
        path = os.path.join(root_dir, f"{prefix}_newsapi_content.json")
    elif provider == "alphavantage":
        path = os.path.join(root_dir, f"{prefix}_alphavantage_content.json")
    elif provider == "forexnewsapi":
        path = os.path.join(root_dir, f"{prefix}_forexnewsapi_content.json")
    elif provider == "eodhd":
        path = os.path.join(root_dir, f"{prefix}_eodhd_content.json")
    else:
        raise ValueError(f"Unknown provider {provider!r}")

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _article_datetime(article: Dict[str, Any]) -> Optional[datetime]:
    """
    Accepts either:
    - cached dicts: "publishedAt" ISO string
    - normalized items: "published_at" ISO string
    """
    if not isinstance(article, dict):
        return None
    v = article.get("publishedAt") or article.get("published_at") or article.get("publishedAt".lower())
    if isinstance(v, str) and v.strip():
        return nf.parse_iso_like_datetime(v)
    return None


def _article_url(article: Dict[str, Any]) -> str:
    if not isinstance(article, dict):
        return ""
    u = article.get("url") or article.get("link") or ""
    return u.strip() if isinstance(u, str) else ""


def _article_title(article: Dict[str, Any]) -> str:
    if not isinstance(article, dict):
        return ""
    t = article.get("title") or ""
    return t.strip() if isinstance(t, str) else ""


def _article_snippet(article: Dict[str, Any]) -> str:
    if not isinstance(article, dict):
        return ""
    # cached stores use "content"; normalized items use "snippet" or "content"
    s = article.get("snippet") or article.get("content") or article.get("description") or ""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return s[:800]


def _dedupe_by_url(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for a in articles:
        url = _article_url(a)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(a)
    return out


def filter_articles_by_window(
    articles: List[Dict[str, Any]],
    *,
    start_date: Optional[date],
    end_date: Optional[date],
) -> List[Dict[str, Any]]:
    if start_date is None and end_date is None:
        return articles
    kept: List[Dict[str, Any]] = []
    for a in articles:
        dt = _article_datetime(a)
        if nf.within_date_range(dt, start_date, end_date):
            kept.append(a)
    return kept


# -----------------------------------------------------------------------------
# Fetch wrappers (live API calls)
# -----------------------------------------------------------------------------


def fetch_provider_articles_live(
    *,
    pair: str,
    provider: str,
    start_date: Optional[date],
    end_date: Optional[date],
    newsapi_max_items: int,
    forexnewsapi_max_items: int,
    alphavantage_max_items: int,
    eodhd_limit: int,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts in a common schema:
    - url, title, publishedAt, content, source
    """
    provider = provider.lower().strip()
    if provider == "newsapi":
        items = nf.fetch_newsapi_fx_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_items=newsapi_max_items,
        )
        out: List[Dict[str, Any]] = []
        for it in items:
            url = it.get("url")
            title = it.get("title")
            dt = it.get("published_dt")
            snippet = it.get("snippet") or ""
            source = it.get("source") or "NewsAPI"
            if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
                continue
            out.append(
                {
                    "url": url.strip(),
                    "title": title.strip(),
                    "publishedAt": dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else "",
                    "content": snippet,
                    "source": source,
                }
            )
        return out

    if provider == "alphavantage":
        items = nf.fetch_alphavantage_fx_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_items=alphavantage_max_items,
        )
        out = []
        for it in items:
            url = it.get("url")
            title = it.get("title")
            dt = it.get("published_dt")
            snippet = it.get("snippet") or ""
            source = it.get("source") or "AlphaVantage"
            if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
                continue
            out.append(
                {
                    "url": url.strip(),
                    "title": title.strip(),
                    "publishedAt": dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else "",
                    "content": snippet,
                    "source": source,
                }
            )
        return out

    if provider == "forexnewsapi":
        items = nf.fetch_forexnewsapi_fx_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            max_items=forexnewsapi_max_items,
        )
        out = []
        for it in items:
            url = it.get("url")
            title = it.get("title")
            dt = it.get("published_dt")
            snippet = it.get("snippet") or ""
            source = it.get("source") or "ForexNewsAPI"
            if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
                continue
            out.append(
                {
                    "url": url.strip(),
                    "title": title.strip(),
                    "publishedAt": dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else "",
                    "content": snippet,
                    "source": source,
                }
            )
        return out

    if provider == "eodhd":
        symbol = nf._eodhd_symbol_for_fx_pair(pair)
        items = nf.fetch_eodhd_news(
            asset_label=pair,
            eodhd_symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            limit=eodhd_limit,
            offset=0,
        )
        out = []
        for it in items:
            url = it.get("url")
            title = it.get("title")
            dt = it.get("published_dt")
            snippet = it.get("snippet") or ""
            source = it.get("source") or "EODHD"
            if not (isinstance(url, str) and url.strip() and isinstance(title, str) and title.strip()):
                continue
            out.append(
                {
                    "url": url.strip(),
                    "title": title.strip(),
                    "publishedAt": dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else "",
                    "content": snippet,
                    "source": source,
                }
            )
        return out

    raise ValueError(f"Unknown provider {provider!r}")


# -----------------------------------------------------------------------------
# Metrics + reporting
# -----------------------------------------------------------------------------


def _minutes_since(dt: datetime, now: datetime) -> float:
    return max(0.0, (now - dt).total_seconds() / 60.0)


def summarize_provider(
    *,
    pair: str,
    provider_name: str,
    articles: List[Dict[str, Any]],
    now_utc: datetime,
    quality_threshold: int,
) -> Dict[str, Any]:
    articles = _dedupe_by_url(articles)

    dts: List[datetime] = []
    hosts: List[str] = []
    scored: List[Dict[str, Any]] = []

    for a in articles:
        url = _article_url(a)
        title = _article_title(a)
        snippet = _article_snippet(a)
        dt = _article_datetime(a)
        if isinstance(dt, datetime):
            dts.append(dt)
        h = host_from_url(url)
        if h:
            hosts.append(h)

        rs = rule_score_article(pair=pair, title=title, snippet=snippet, url=url)
        scored.append(
            {
                "url": url,
                "title": title,
                "publishedAt": dt.astimezone(timezone.utc).isoformat() if isinstance(dt, datetime) else None,
                "host": h or None,
                "rule_quality_score": rs.score_0_10,
                "reliability_tier": rs.reliability_tier,
                "reasons": rs.reasons,
            }
        )

    most_recent_dt = max(dts) if dts else None
    oldest_dt = min(dts) if dts else None

    # Age buckets (helps compare "how many recent items" by provider).
    age_24h = 0
    age_3d = 0
    age_7d = 0
    for dt in dts:
        mins = _minutes_since(dt, now_utc)
        if mins <= 24 * 60:
            age_24h += 1
        if mins <= 3 * 24 * 60:
            age_3d += 1
        if mins <= 7 * 24 * 60:
            age_7d += 1

    high_quality = [
        x
        for x in scored
        if isinstance(x.get("rule_quality_score"), int)
        and x["rule_quality_score"] >= quality_threshold
        and x.get("reliability_tier") != "low"
    ]
    high_quality_sorted = sorted(
        high_quality,
        key=lambda x: (
            int(x.get("rule_quality_score") or 0),
            1 if x.get("reliability_tier") == "tier_1" else 0,
        ),
        reverse=True,
    )

    # "Most recent correlated FX financial news" proxy: newest among rule-qualified items.
    hq_dts: List[datetime] = []
    for x in high_quality:
        pa = x.get("publishedAt")
        if isinstance(pa, str) and pa.strip():
            dt = nf.parse_iso_like_datetime(pa)
            if isinstance(dt, datetime):
                hq_dts.append(dt)
    hq_most_recent_dt = max(hq_dts) if hq_dts else None

    return {
        "provider": provider_name,
        "items": len(articles),
        "items_with_time": len(dts),
        "unique_hosts": len(set(hosts)),
        "most_recent_published_at": most_recent_dt.astimezone(timezone.utc).isoformat()
        if isinstance(most_recent_dt, datetime)
        else None,
        "recency_lag_minutes": round(_minutes_since(most_recent_dt, now_utc), 1)
        if isinstance(most_recent_dt, datetime)
        else None,
        "recent_counts": {"<=24h": age_24h, "<=3d": age_3d, "<=7d": age_7d},
        "oldest_published_at": oldest_dt.astimezone(timezone.utc).isoformat()
        if isinstance(oldest_dt, datetime)
        else None,
        "high_quality_count": len(high_quality),
        "most_recent_high_quality_published_at": hq_most_recent_dt.astimezone(timezone.utc).isoformat()
        if isinstance(hq_most_recent_dt, datetime)
        else None,
        "high_quality_recency_lag_minutes": round(_minutes_since(hq_most_recent_dt, now_utc), 1)
        if isinstance(hq_most_recent_dt, datetime)
        else None,
        "high_quality_top_urls": [x["url"] for x in high_quality_sorted[:5] if isinstance(x.get("url"), str)],
        "scored_items": scored,
    }


def _print_pair_header(pair: str) -> None:
    print("\n" + "=" * 88)
    print(f"[PAIR] {pair}")
    print("=" * 88)


def _format_lag(lag_min: Optional[float]) -> str:
    if lag_min is None:
        return "n/a"
    if lag_min < 90:
        return f"{lag_min:.0f}m"
    hours = lag_min / 60.0
    if hours < 48:
        return f"{hours:.1f}h"
    days = hours / 24.0
    return f"{days:.1f}d"


def _to_local_iso(iso_utc: Optional[str]) -> Optional[str]:
    """
    Convert an ISO timestamp (expected UTC or offset-aware) into local timezone ISO.
    If input is missing/invalid, returns None.
    """
    if not (isinstance(iso_utc, str) and iso_utc.strip()):
        return None
    dt = nf.parse_iso_like_datetime(iso_utc)
    if not isinstance(dt, datetime):
        return None
    try:
        return dt.astimezone().isoformat()
    except Exception:
        return None


def print_provider_summary_row(summary: Dict[str, Any], *, quality_threshold: int) -> None:
    prov = summary.get("provider", "Unknown")
    items = summary.get("items", 0)
    lag = _format_lag(summary.get("recency_lag_minutes"))
    hq_lag = _format_lag(summary.get("high_quality_recency_lag_minutes"))
    recent = summary.get("most_recent_published_at") or "n/a"
    hq = summary.get("high_quality_count", 0)
    hosts = summary.get("unique_hosts", 0)
    print(
        f"- {prov:12s} | items={items:3d} | newest_lag={lag:>5s} | HQ_newest_lag={hq_lag:>5s} "
        f"| newest={recent} | hosts={hosts:3d} | HQ(>= {quality_threshold})={hq:3d}"
    )


def best_recency_across_providers(provider_summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_lag: Optional[float] = None
    for s in provider_summaries:
        # Prefer newest among *high-quality* items; fall back to any item.
        lag_val: Any = s.get("high_quality_recency_lag_minutes")
        if not isinstance(lag_val, (int, float)):
            lag_val = s.get("recency_lag_minutes")
        if not isinstance(lag_val, (int, float)):
            continue
        lag = float(lag_val)
        if best is None or best_lag is None or lag < best_lag:
            best = s
            best_lag = lag
    return best


def _deepseek_provider_label(provider: str) -> str:
    p = provider.lower()
    if p == "newsapi":
        return "NewsAPI"
    if p == "alphavantage":
        return "AlphaVantage"
    if p == "forexnewsapi":
        return "ForexNewsAPI"
    if p == "eodhd":
        return "EODHD"
    return provider


def maybe_run_deepseek(
    *,
    pair: str,
    provider: str,
    summary: Dict[str, Any],
    max_articles: int,
    quality_threshold: int,
) -> Optional[Dict[str, Any]]:
    """
    Run DeepSeek relevance/quality *after* rule filtering, to reduce noise/cost.
    Returns DeepSeek payload (or None if skipped/error).
    """
    scored = summary.get("scored_items")
    if not isinstance(scored, list) or not scored:
        return None

    candidates = [
        x
        for x in scored
        if isinstance(x, dict)
        and isinstance(x.get("url"), str)
        and isinstance(x.get("title"), str)
        and isinstance(x.get("rule_quality_score"), int)
        and x["rule_quality_score"] >= quality_threshold
        and x.get("reliability_tier") != "low"
    ]
    # Keep the highest-scoring candidates.
    candidates = sorted(
        candidates,
        key=lambda x: int(x.get("rule_quality_score") or 0),
        reverse=True,
    )[:max_articles]

    articles_for_llm: List[Dict[str, Any]] = []
    for c in candidates:
        articles_for_llm.append(
            {
                "url": c.get("url"),
                "title": c.get("title"),
                "publishedAt": c.get("publishedAt") or "",
                "content": _article_snippet(c)[:500],
                "source": c.get("host") or "",
            }
        )

    source_name = _deepseek_provider_label(provider)
    payload = nf.run_deepseek_source_relevance(
        asset=pair,
        source_name=source_name,
        articles=articles_for_llm,
    )
    return payload if isinstance(payload, dict) else None


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark API FX news sources (NewsAPI / AlphaVantage / EODHD).")
    p.add_argument(
        "--pairs",
        type=str,
        default=None,
        help='Comma-separated FX pairs. If omitted, uses `benchmark_fx_pairs` from default_config.json.',
    )
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD (UTC, inclusive).")
    p.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD (UTC, inclusive).")
    p.add_argument("--window-days", type=int, default=7, help="Lookback window (days) if start/end not provided.")
    p.add_argument("--from-cache", action="store_true", help="Use local append-only content stores (no API calls).")
    p.add_argument("--cache-root", type=str, default=".", help="Directory containing *_content.json files.")

    p.add_argument("--newsapi-max", type=int, default=100, help="Max items per NewsAPI call (<=100).")
    p.add_argument("--forexnewsapi-max", type=int, default=3, help="Max items per ForexNewsAPI call (trial plans may require <=3).")
    p.add_argument("--alphavantage-max", type=int, default=50, help="Max FX items kept after AlphaVantage filtering.")
    p.add_argument("--eodhd-limit", type=int, default=100, help="EODHD limit parameter.")

    p.add_argument("--quality-threshold", type=int, default=7, help="Rule-based high quality threshold (0-10).")
    p.add_argument("--deepseek", action="store_true", help="Run DeepSeek relevance/quality after rule filtering.")
    p.add_argument("--deepseek-max-articles", type=int, default=15, help="Max candidate articles sent to DeepSeek per provider.")

    p.add_argument("--log-jsonl", type=str, default=DEFAULT_LOG_JSONL, help=f"Append results to a JSONL log file (default: {DEFAULT_LOG_JSONL}).")
    p.add_argument("--no-log", action="store_true", help="Disable JSONL logging.")
    p.add_argument("--verbose", action="store_true", help="Verbose warnings (reuses news_fetching VERBOSE).")
    return p.parse_args()


def _pairs_from_default_config(cfg: Dict[str, Any]) -> List[str]:
    """
    Reads `benchmark_fx_pairs` from default_config.json.
    Allowed shapes:
    - "all" (case-insensitive): run all supported FX pairs
    - "EUR/USD" (or any single pair): run only that pair
    - ["EUR/USD", "USD/JPY", ...]: run a list
    Falls back to DEFAULT_FX_PAIRS if unset/invalid.
    """
    def _norm(x: str) -> str:
        s = x.strip()
        m = re.match(r"^([A-Za-z]{3})[_/ ]([A-Za-z]{3})$", s)
        if m:
            return f"{m.group(1).upper()}/{m.group(2).upper()}"
        return s

    raw = cfg.get("benchmark_fx_pairs")
    if isinstance(raw, str):
        val = _norm(raw)
        if not val:
            # Fall back to the main configured pair if present.
            cp = cfg.get("currency_pair")
            if isinstance(cp, str) and cp.strip():
                return [_norm(cp)]
            return DEFAULT_FX_PAIRS[:]
        if val.lower() == "all":
            return DEFAULT_FX_PAIRS[:]
        return [val]
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if isinstance(x, str) and x.strip():
                out.append(_norm(x))
        return out if out else DEFAULT_FX_PAIRS[:]
    cp = cfg.get("currency_pair")
    if isinstance(cp, str) and cp.strip():
        return [_norm(cp)]
    return DEFAULT_FX_PAIRS[:]


def _load_default_config_raw(path: str) -> Dict[str, Any]:
    """
    Load default_config.json without schema-normalization so custom keys
    (like `benchmark_fx_pairs`) are preserved.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _print_collecting_times(provider_summaries: List[Dict[str, Any]]) -> None:
    print("\n[COLLECTING TIMES] (most recent high-quality item per provider; fallback to newest if none)")
    for s in provider_summaries:
        prov = str(s.get("provider") or "Unknown")
        hq_time = s.get("most_recent_high_quality_published_at")
        hq_lag = s.get("high_quality_recency_lag_minutes")
        if isinstance(hq_time, str) and hq_time.strip() and isinstance(hq_lag, (int, float)):
            hq_local = _to_local_iso(hq_time) or "n/a"
            print(f"- {prov:12s}: HQ_time_utc={hq_time} | HQ_time_local={hq_local} | lag={_format_lag(hq_lag)}")
            continue
        t = s.get("most_recent_published_at")
        lag = s.get("recency_lag_minutes")
        if isinstance(t, str) and t.strip() and isinstance(lag, (int, float)):
            t_local = _to_local_iso(t) or "n/a"
            print(f"- {prov:12s}: time_utc={t} | time_local={t_local} | lag={_format_lag(lag)}")
        else:
            print(f"- {prov:12s}: n/a")


def _print_raw_collecting_times(provider_summaries: List[Dict[str, Any]]) -> None:
    print("\n[RAW COLLECTING TIMES] (newest item per provider, regardless of quality)")
    for s in provider_summaries:
        prov = str(s.get("provider") or "Unknown")
        t = s.get("most_recent_published_at")
        lag = s.get("recency_lag_minutes")
        if isinstance(t, str) and t.strip() and isinstance(lag, (int, float)):
            t_local = _to_local_iso(t) or "n/a"
            print(f"- {prov:12s}: time_utc={t} | time_local={t_local} | lag={_format_lag(lag)}")
        else:
            print(f"- {prov:12s}: n/a")


def best_raw_recency_across_providers(provider_summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_lag: Optional[float] = None
    for s in provider_summaries:
        lag_val: Any = s.get("recency_lag_minutes")
        if not isinstance(lag_val, (int, float)):
            continue
        lag = float(lag_val)
        if best is None or best_lag is None or lag < best_lag:
            best = s
            best_lag = lag
    return best


def main() -> None:
    args = parse_args()
    load_dotenv()
    nf.VERBOSE = bool(args.verbose)

    cfg = _load_default_config_raw(getattr(nf, "DEFAULT_CONFIG_PATH", "default_config.json"))

    if isinstance(args.pairs, str) and args.pairs.strip():
        def _norm_cli(x: str) -> str:
            s = x.strip()
            m = re.match(r"^([A-Za-z]{3})[_/ ]([A-Za-z]{3})$", s)
            if m:
                return f"{m.group(1).upper()}/{m.group(2).upper()}"
            return s

        pairs = [_norm_cli(x) for x in args.pairs.split(",") if x.strip()]
    else:
        pairs = _pairs_from_default_config(cfg)

    # Ensure EUR/USD first when present.
    if "EUR/USD" in pairs:
        pairs = ["EUR/USD"] + [p for p in pairs if p != "EUR/USD"]

    start_date = nf.parse_cli_date(args.start_date)
    end_date = nf.parse_cli_date(args.end_date)
    if start_date is None and end_date is None:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=max(1, int(args.window_days)))

    now_utc = datetime.now(timezone.utc)
    now_local = datetime.now().astimezone()
    print(f"[RUN] run_at_utc={now_utc.isoformat()} | run_at_local={now_local.isoformat()}")

    providers = ["newsapi", "forexnewsapi", "alphavantage", "eodhd"]

    all_results: Dict[str, Any] = {
        "schema_version": 1,
        "run_at": now_utc.isoformat(),
        "run_at_local": now_local.isoformat(),
        "window": {"start_date": start_date.isoformat() if start_date else None, "end_date": end_date.isoformat() if end_date else None},
        "from_cache": bool(args.from_cache),
        "pairs": {},
    }

    for pair in pairs:
        _print_pair_header(pair)
        provider_summaries: List[Dict[str, Any]] = []
        deepseek_outputs: Dict[str, Any] = {}

        for prov in providers:
            if args.from_cache:
                articles = load_cached_provider_articles(pair=pair, provider=prov, root_dir=args.cache_root)
                articles = filter_articles_by_window(articles, start_date=start_date, end_date=end_date)
            else:
                articles = fetch_provider_articles_live(
                    pair=pair,
                    provider=prov,
                    start_date=start_date,
                    end_date=end_date,
                    newsapi_max_items=int(args.newsapi_max),
                    forexnewsapi_max_items=int(args.forexnewsapi_max),
                    alphavantage_max_items=int(args.alphavantage_max),
                    eodhd_limit=int(args.eodhd_limit),
                )

            summary = summarize_provider(
                pair=pair,
                provider_name=_deepseek_provider_label(prov),
                articles=articles,
                now_utc=now_utc,
                quality_threshold=int(args.quality_threshold),
            )
            provider_summaries.append(summary)
            print_provider_summary_row(summary, quality_threshold=int(args.quality_threshold))

            if args.deepseek:
                ds = maybe_run_deepseek(
                    pair=pair,
                    provider=prov,
                    summary=summary,
                    max_articles=int(args.deepseek_max_articles),
                    quality_threshold=int(args.quality_threshold),
                )
                if isinstance(ds, dict):
                    deepseek_outputs[_deepseek_provider_label(prov)] = ds
                    counts = ds.get("counts")
                    if isinstance(counts, dict):
                        mr = counts.get("most_relevant", 0)
                        r = counts.get("relevant", 0)
                        u = counts.get("unrelated", 0)
                        print(f"  DeepSeek: most_relevant={mr} relevant={r} unrelated={u}")

        _print_collecting_times(provider_summaries)
        _print_raw_collecting_times(provider_summaries)

        best = best_recency_across_providers(provider_summaries)
        if best is not None:
            best_time = best.get("most_recent_high_quality_published_at") or best.get("most_recent_published_at")
            best_lag = best.get("high_quality_recency_lag_minutes")
            best_kind = "HQ newest" if isinstance(best_lag, (int, float)) else "newest"
            if not isinstance(best_lag, (int, float)):
                best_lag = best.get("recency_lag_minutes")
            print(
                f"\n[RECENCY] Best {best_kind} article right now: {best.get('provider')} "
                f"(lag={_format_lag(best_lag)}, time={best_time})"
            )
        else:
            print("\n[RECENCY] Best newest article right now: n/a (no publish times parsed).")

        best_raw = best_raw_recency_across_providers(provider_summaries)
        if best_raw is not None:
            print(
                f"[RECENCY] Best RAW newest article right now: {best_raw.get('provider')} "
                f"(lag={_format_lag(best_raw.get('recency_lag_minutes'))}, time={best_raw.get('most_recent_published_at')})"
            )

        all_results["pairs"][pair] = {
            "providers": provider_summaries,
            "deepseek": deepseek_outputs,
        }

    if not bool(args.no_log) and isinstance(args.log_jsonl, str) and args.log_jsonl.strip():
        append_jsonl(args.log_jsonl.strip(), all_results)
        print(f"\n[INFO] Appended benchmark results to {args.log_jsonl.strip()!r}")


if __name__ == "__main__":
    main()

