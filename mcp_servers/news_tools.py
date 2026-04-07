"""
News tool functions used by both the MCP server and the orchestrator.
Data source: Google News RSS feed (no API key required).
"""

import feedparser
import httpx
import re
from datetime import datetime
from typing import Optional

GOOGLE_NEWS_RSS = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_SEARCH = (
    "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
)

# Spoof a browser User-Agent — Google News RSS requires it
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}


def _strip_html(text: str) -> str:
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def _parse_published(entry) -> str:
    """Return a human-readable publication date from a feed entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            dt = datetime(*entry.published_parsed[:6])
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            pass
    return entry.get("published", "Unknown date")


def get_latest_news(
    query: Optional[str] = None,
    count: int = 8,
) -> dict:
    """
    Fetch latest news articles from Google News RSS.

    Args:
        query: Optional keyword / phrase to search for. If None, returns top headlines.
        count: Maximum number of articles to return (1–20).

    Returns:
        Structured dict with article list and metadata.
    """
    count = max(1, min(count, 20))

    if query and query.strip():
        encoded = query.strip().replace(" ", "+")
        url = GOOGLE_NEWS_SEARCH.format(query=encoded)
        feed_label = f'search: "{query}"'
    else:
        url = GOOGLE_NEWS_RSS
        feed_label = "top headlines"

    # Use httpx to fetch the raw XML so we control SSL/TLS and headers,
    # then pass the content string to feedparser for parsing.
    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        feed = feedparser.parse(resp.text)
    except Exception as exc:
        return {
            "query": feed_label,
            "fetched_count": 0,
            "articles": [],
            "feed_title": "Google News",
            "source_url": url,
            "data_source": "Google News RSS (https://news.google.com/rss)",
            "error": str(exc),
        }

    articles = []
    for entry in feed.entries[:count]:
        source_title = "Unknown"
        if hasattr(entry, "source") and isinstance(entry.source, dict):
            source_title = entry.source.get("title", "Unknown")
        elif hasattr(entry, "source"):
            source_title = str(entry.source)

        summary = _strip_html(entry.get("summary", ""))
        if len(summary) > 400:
            summary = summary[:397] + "..."

        articles.append(
            {
                "title": _strip_html(entry.get("title", "No title")),
                "source": source_title,
                "published": _parse_published(entry),
                "summary": summary,
                "link": entry.get("link", ""),
            }
        )

    return {
        "query": feed_label,
        "fetched_count": len(articles),
        "articles": articles,
        "feed_title": feed.feed.get("title", "Google News"),
        "source_url": url,
        "data_source": "Google News RSS (https://news.google.com/rss)",
    }
