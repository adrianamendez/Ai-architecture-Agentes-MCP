"""
Synchronous bridge to the news-aggregator-mcp-server pip package.

The package exposes its tools as async functions; this module wraps them
in synchronous calls so the orchestrator (running in Streamlit's main thread)
can invoke them without needing an event loop.

Available tools (11 total):
  RSS/Atom   : fetch_feed, get_news_by_category, search_rss_feeds, list_feed_catalog
  HackerNews : get_hackernews_top, get_hackernews_story, get_hackernews_trending
  GDELT      : search_global_news, get_news_timeline, get_news_by_country, get_trending_topics
"""

import asyncio
from typing import Any

# Importamos los helpers internos del paquete instalado (src/tools/*)
from src.tools.rss import (
    _feed_abrufen,
    _artikel_extrahieren,
    FEED_KATALOG,
)
from src.tools.hackernews import _top_stories_laden, _item_abrufen
from src.tools.gdelt import _gdelt_suche, _artikel_formatieren

import httpx


def _normalize_gdelt_article(a: dict) -> dict:
    """Converts German-keyed GDELT article dicts to English keys."""
    return {
        "title": a.get("titel", a.get("title", "")),
        "url": a.get("url", ""),
        "source": a.get("quelle", a.get("source", "")),
        "language": a.get("sprache", a.get("language", "")),
        "country": a.get("land", a.get("country", "")),
        "seen_date": a.get("seendate", ""),
        "social_shares": a.get("socialshares", 0),
    }


def _normalize_story(s: dict) -> dict:
    """Converts German-keyed HackerNews story dicts to English keys."""
    normalized = {
        "id": s.get("id"),
        "title": s.get("titel", s.get("title", "")),
        "url": s.get("url", ""),
        "points": s.get("punkte", s.get("points", 0)),
        "comments": s.get("kommentare", s.get("comments", 0)),
        "author": s.get("autor", s.get("author", "")),
        "published_unix": s.get("veroeffentlicht_unix"),
        "hn_link": s.get("hn_link", ""),
        "type": s.get("typ", s.get("type", "story")),
    }
    if "matched_keyword" in s:
        normalized["matched_keyword"] = s["matched_keyword"]
    return normalized


def _normalize_article(a: dict) -> dict:
    """Converts German-keyed RSS article dicts to English keys."""
    return {
        "title": a.get("titel", a.get("title", "")),
        "url": a.get("url", ""),
        "summary": a.get("zusammenfassung", a.get("summary", "")),
        "published": a.get("veroeffentlicht", a.get("published", "")),
        "author": a.get("autor", a.get("author", "")),
        "tags": a.get("tags", []),
        "source": a.get("source", ""),
        "category": a.get("category", ""),
    }


def _run(coro) -> Any:
    """Runs a coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Streamlit may already have a running loop; use a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# RSS / Atom Tools
# ---------------------------------------------------------------------------

def fetch_feed(feed_url: str, max_articles: int = 10) -> dict:
    """Fetches any RSS/Atom feed and returns structured articles.

    Args:
        feed_url: RSS/Atom feed URL.
        max_articles: Maximum number of articles (1-50).
    """
    max_articles = min(max_articles, 50)
    return _run(_feed_abrufen(feed_url, max_articles))


def get_news_by_category(category: str, max_per_feed: int = 3) -> dict:
    """Aggregates news from predefined sources by category.

    Available categories: tech, ai, general, business, crypto, science.

    Args:
        category: News category.
        max_per_feed: Articles per source (default 3).
    """
    category = category.lower().strip()
    if category not in FEED_KATALOG:
        return {
            "error": f"Unknown category '{category}'",
            "available_categories": list(FEED_KATALOG.keys()),
        }

    async def _gather():
        feeds = FEED_KATALOG[category]
        tasks = [_feed_abrufen(f["url"], max_per_feed) for f in feeds]
        results = await asyncio.gather(*tasks)
        all_articles = []
        sources = []
        for i, res in enumerate(results):
            if res.get("status") == "ok":
                sources.append({"name": feeds[i]["name"], "url": feeds[i]["url"]})
                for art in res.get("artikel", []):
                    art["source"] = feeds[i]["name"]
                    all_articles.append(_normalize_article(art))
        return {
            "category": category,
            "sources": sources,
            "total_articles": len(all_articles),
            "articles": all_articles,
        }

    return _run(_gather())


def search_rss_feeds(query: str, categories: str = "all", max_results: int = 10) -> dict:
    """Searches news across all RSS feeds by keyword.

    Args:
        query: Search term (e.g. 'AI regulation', 'Bitcoin').
        categories: Comma-separated categories or 'all'.
        max_results: Maximum number of results.
    """
    query_lower = query.lower()

    if categories.lower() == "all":
        active_cats = list(FEED_KATALOG.keys())
    else:
        active_cats = [c.strip().lower() for c in categories.split(",") if c.strip().lower() in FEED_KATALOG]

    if not active_cats:
        return {"error": "No valid categories", "available": list(FEED_KATALOG.keys())}

    async def _gather():
        all_feeds = []
        feed_to_cat = {}
        for cat in active_cats:
            for fi in FEED_KATALOG[cat]:
                all_feeds.append(fi)
                feed_to_cat[fi["url"]] = cat

        results = await asyncio.gather(*[_feed_abrufen(f["url"], 10) for f in all_feeds])

        hits = []
        for i, res in enumerate(results):
            if res.get("status") != "ok":
                continue
            source = all_feeds[i]["name"]
            cat = feed_to_cat[all_feeds[i]["url"]]
            for art in res.get("artikel", []):
                title_l = art.get("titel", "").lower()
                summary_l = art.get("zusammenfassung", "").lower()
                if query_lower in title_l or query_lower in summary_l:
                    hits.append(_normalize_article({**art, "source": source, "category": cat}))

        hits.sort(key=lambda a: 2 if query_lower in a.get("titel", "").lower() else 1, reverse=True)
        return {
            "query": query,
            "categories_searched": active_cats,
            "total_hits": len(hits),
            "articles": hits[:max_results],
        }

    return _run(_gather())


def list_feed_catalog() -> dict:
    """Lists the full catalog of predefined RSS feeds."""
    catalog = {cat: [{"name": f["name"], "url": f["url"]} for f in feeds]
               for cat, feeds in FEED_KATALOG.items()}
    return {
        "categories": list(FEED_KATALOG.keys()),
        "total_feeds": sum(len(f) for f in FEED_KATALOG.values()),
        "catalog": catalog,
    }


# ---------------------------------------------------------------------------
# HackerNews Tools
# ---------------------------------------------------------------------------

def get_hackernews_top(story_type: str = "top", limit: int = 10) -> dict:
    """Fetches top stories from Hacker News.

    Args:
        story_type: Type — top, new, best, ask, show, jobs.
        limit: Number of stories (max 30).
    """
    limit = min(limit, 30)
    stories = _run(_top_stories_laden(story_type, limit))
    stories = [_normalize_story(s) for s in stories]
    return {"type": story_type, "count": len(stories), "stories": stories, "source": "Hacker News"}


def get_hackernews_story(story_id: int) -> dict:
    """Fetches full details of a specific HackerNews story.

    Args:
        story_id: Numeric HN story ID.
    """
    async def _fetch():
        async with httpx.AsyncClient(timeout=10.0) as client:
            item = await _item_abrufen(client, story_id)
        if not item:
            return {"error": f"Story {story_id} not found"}
        return {
            "id": item.get("id"),
            "title": item.get("title", ""),
            "url": item.get("url", f"https://news.ycombinator.com/item?id={story_id}"),
            "score": item.get("score", 0),
            "comments": item.get("descendants", 0),
            "author": item.get("by", ""),
            "hn_link": f"https://news.ycombinator.com/item?id={story_id}",
        }

    return _run(_fetch())


def get_hackernews_trending(keywords: str, limit: int = 5) -> dict:
    """Searches HackerNews stories that contain the given keywords.

    Args:
        keywords: Comma-separated keywords (e.g. 'AI, Claude, MCP').
        limit: Maximum number of results.
    """
    async def _search():
        terms = [k.strip().lower() for k in keywords.split(",")]
        stories = await _top_stories_laden("top", 50)
        hits = []
        for s in stories:
            title_l = s["titel"].lower()
            for term in terms:
                if term in title_l:
                    s["matched_keyword"] = term
                    hits.append(s)
                    break
        return {
            "keywords": terms,
            "matches": [_normalize_story(s) for s in hits[:limit]],
            "match_count": len(hits),
            "searched_in": "HackerNews Top 50",
        }

    return _run(_search())


# ---------------------------------------------------------------------------
# GDELT Global News Tools
# ---------------------------------------------------------------------------

def search_global_news(query: str, max_records: int = 10, language: str = "") -> dict:
    """Searches global news via GDELT (65+ languages, 100+ countries).

    Args:
        query: Search query (e.g. 'AI regulation', 'climate Colombia').
        max_records: Maximum number of results (1-250).
        language: Language filter (e.g. 'english', 'spanish'). Empty = all.
    """
    max_records = min(max_records, 250)
    search_query = f"{query} sourcelang:{language}" if language else query
    params = {
        "query": search_query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "DateDesc",
    }

    try:
        data = _run(_gdelt_suche(params))
    except Exception as exc:
        return {"error": f"GDELT request failed: {exc}", "query": query}

    if "error" in data:
        return {"error": data["error"], "query": query}

    articles = data.get("articles", [])
    return {
        "query": query,
        "language_filter": language or "all",
        "found": len(articles),
        "articles": [_normalize_gdelt_article(_artikel_formatieren(a)) for a in articles],
        "source": "GDELT Project (Global News Database)",
    }


def get_news_timeline(query: str, timespan: str = "1d") -> dict:
    """Analyzes news volume over time for a topic (trend analysis).

    Args:
        query: Topic or search term.
        timespan: Period — 15min, 1h, 4h, 1d, 3d, 7d, 1m.
    """
    valid = ["15min", "1h", "4h", "1d", "3d", "7d", "1m"]
    if timespan not in valid:
        timespan = "1d"

    params = {"query": query, "mode": "timelinevol", "timespan": timespan, "format": "json"}
    try:
        data = _run(_gdelt_suche(params))
    except Exception as exc:
        return {"error": f"GDELT request failed: {exc}", "query": query}

    if "error" in data:
        return {"error": data["error"], "query": query}

    timeline_raw = data.get("timeline", [{}])
    points = timeline_raw[0].get("data", [])[-20:] if timeline_raw else []
    timeline = [{"time": p.get("date", ""), "volume": p.get("value", 0)} for p in points]

    trend = "unknown"
    if len(timeline) >= 2:
        first, last = timeline[0]["volume"], timeline[-1]["volume"]
        trend = "rising" if last > first * 1.2 else ("falling" if last < first * 0.8 else "stable")

    return {
        "query": query,
        "timespan": timespan,
        "trend": trend,
        "data_points": len(timeline),
        "timeline": timeline,
        "source": "GDELT News Volume Analysis",
    }


def get_news_by_country(country_code: str, query: str = "", max_records: int = 10) -> dict:
    """Fetches news from a specific country via GDELT.

    Args:
        country_code: 2-letter ISO country code (e.g. CO, US, DE, GB, FR).
        query: Additional search term (optional).
        max_records: Maximum number of results.
    """
    country_code = country_code.upper()
    search_query = f"{query} sourcecountry:{country_code}" if query else f"sourcecountry:{country_code}"
    params = {
        "query": search_query,
        "mode": "artlist",
        "maxrecords": max_records,
        "format": "json",
        "sort": "DateDesc",
    }
    try:
        data = _run(_gdelt_suche(params))
    except Exception as exc:
        return {"error": f"GDELT request failed: {exc}", "country": country_code}

    if "error" in data:
        return {"error": data["error"], "country": country_code}

    articles = data.get("articles", [])
    return {
        "country": country_code,
        "query": query or "(all topics)",
        "found": len(articles),
        "articles": [_normalize_gdelt_article(_artikel_formatieren(a)) for a in articles],
        "source": "GDELT Global News",
    }


def get_trending_topics(timespan: str = "1d", tone_filter: str = "") -> dict:
    """Identifies globally trending topics via GDELT.

    Args:
        timespan: Period — 1h, 4h, 1d, 3d, 7d.
        tone_filter: Sentiment filter — positive, negative, neutral. Empty = all.
    """
    tone_suffix = ""
    if tone_filter == "positive":
        tone_suffix = " tone>5"
    elif tone_filter == "negative":
        tone_suffix = " tone<-5"

    themes = [
        "AI artificial intelligence",
        "economy finance market",
        "climate environment",
        "politics government",
        "technology startup",
    ]

    async def _gather_themes():
        results = []
        async with httpx.AsyncClient(timeout=15.0) as client:
            for theme in themes:
                try:
                    resp = await client.get(
                        "https://api.gdeltproject.org/api/v2/doc/doc",
                        params={
                            "query": theme + tone_suffix,
                            "mode": "artlist",
                            "maxrecords": 3,
                            "timespan": timespan,
                            "format": "json",
                        },
                    )
                    data = resp.json()
                    arts = data.get("articles", [])
                    if arts:
                        results.append({
                            "theme": theme,
                            "article_count": len(arts),
                            "top_articles": [_normalize_gdelt_article(_artikel_formatieren(a)) for a in arts[:2]],
                        })
                except Exception:
                    continue
        return {
            "timespan": timespan,
            "tone_filter": tone_filter or "all",
            "themes": results,
            "source": "GDELT Global News Intelligence",
        }

    return _run(_gather_themes())


# ---------------------------------------------------------------------------
# Catalog of all available tool functions (for use in orchestrator)
# ---------------------------------------------------------------------------

ALL_NEWS_TOOLS = {
    "fetch_feed": fetch_feed,
    "get_news_by_category": get_news_by_category,
    "search_rss_feeds": search_rss_feeds,
    "list_feed_catalog": list_feed_catalog,
    "get_hackernews_top": get_hackernews_top,
    "get_hackernews_story": get_hackernews_story,
    "get_hackernews_trending": get_hackernews_trending,
    "search_global_news": search_global_news,
    "get_news_timeline": get_news_timeline,
    "get_news_by_country": get_news_by_country,
    "get_trending_topics": get_trending_topics,
}
