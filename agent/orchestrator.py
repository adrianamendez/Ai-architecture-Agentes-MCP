"""
Agent Orchestrator — Claude claude-opus-4-6 with weather and news tools.

Weather tools (custom MCP server):
  - get_current_weather(latitude, longitude, location_name?)
  - resolve_city(city_name)

News tools (news-aggregator-mcp-server, pip):
  RSS/Atom   : fetch_feed, get_news_by_category, search_rss_feeds, list_feed_catalog
  HackerNews : get_hackernews_top, get_hackernews_story, get_hackernews_trending
  GDELT      : search_global_news, get_news_timeline, get_news_by_country, get_trending_topics
"""

import json
from typing import Iterator

import anthropic

from mcp_servers.weather_tools import get_current_weather, resolve_location
from mcp_servers.news_mcp_client import ALL_NEWS_TOOLS

# ---------------------------------------------------------------------------
# Tool schemas for the Claude API
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    # ------------------------------------------------------------------ WEATHER
    {
        "name": "resolve_city",
        "description": (
            "Converts a city name to lat/lon coordinates. "
            "Use this BEFORE get_current_weather when the user mentions a city by name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "Name of the city (e.g. 'bogota', 'london', 'paris').",
                }
            },
            "required": ["city_name"],
        },
    },
    {
        "name": "get_current_weather",
        "description": (
            "Fetches current weather for geographic coordinates using Open-Meteo (free, no API key). "
            "Returns temperature (°C), feels-like, humidity, wind speed, and weather condition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Decimal latitude."},
                "longitude": {"type": "number", "description": "Decimal longitude."},
                "location_name": {"type": "string", "description": "Human-readable place name (optional)."},
            },
            "required": ["latitude", "longitude"],
        },
    },
    # ------------------------------------------------------------------ RSS / ATOM
    {
        "name": "get_news_by_category",
        "description": (
            "Aggregates news from predefined RSS sources by category. "
            "Categories: tech, ai, general, business, crypto, science. "
            "Includes TechCrunch, Wired, BBC, Reuters, Bloomberg, CoinDesk, Nature, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Category: tech | ai | general | business | crypto | science",
                },
                "max_per_feed": {
                    "type": "integer",
                    "description": "Articles per source (default 3).",
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "search_rss_feeds",
        "description": (
            "Searches news across all RSS feeds by keyword. "
            "Useful for specific topics like 'AI regulation', 'Bitcoin', 'climate change'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term."},
                "categories": {
                    "type": "string",
                    "description": "Comma-separated categories or 'all' (default).",
                },
                "max_results": {"type": "integer", "description": "Max results (default 10)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_feed",
        "description": "Fetches any RSS/Atom feed by URL and returns structured articles.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feed_url": {"type": "string", "description": "RSS/Atom feed URL."},
                "max_articles": {"type": "integer", "description": "Max articles (default 10)."},
            },
            "required": ["feed_url"],
        },
    },
    {
        "name": "list_feed_catalog",
        "description": "Lists the full catalog of predefined RSS feeds with their URLs.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    # ------------------------------------------------------------------ HACKERNEWS
    {
        "name": "get_hackernews_top",
        "description": (
            "Fetches top stories from Hacker News. Ideal for technology, "
            "startup, and programming news. story_type: top, new, best, ask, show, jobs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "story_type": {
                    "type": "string",
                    "description": "Type: top | new | best | ask | show | jobs (default: top)",
                },
                "limit": {"type": "integer", "description": "Number of stories (max 30, default 10)."},
            },
            "required": [],
        },
    },
    {
        "name": "get_hackernews_trending",
        "description": "Searches HackerNews stories that contain specific keywords.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Comma-separated keywords (e.g. 'AI, Claude, MCP').",
                },
                "limit": {"type": "integer", "description": "Max results (default 5)."},
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "get_hackernews_story",
        "description": "Fetches full details of a specific HackerNews story by ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "story_id": {"type": "integer", "description": "Numeric HN story ID."},
            },
            "required": ["story_id"],
        },
    },
    # ------------------------------------------------------------------ GDELT
    {
        "name": "search_global_news",
        "description": (
            "Searches global news using GDELT (65+ languages, 100+ countries). "
            "Ideal for international news and global coverage analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (e.g. 'AI regulation', 'Colombia economy')."},
                "max_records": {"type": "integer", "description": "Max results (default 10)."},
                "language": {
                    "type": "string",
                    "description": "Language filter (e.g. 'english', 'spanish'). Empty = all.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_news_by_country",
        "description": "Fetches news from a specific country via GDELT. Uses 2-letter ISO country code.",
        "input_schema": {
            "type": "object",
            "properties": {
                "country_code": {"type": "string", "description": "ISO-2 country code (e.g. CO, US, DE, GB, FR)."},
                "query": {"type": "string", "description": "Additional topic filter (optional)."},
                "max_records": {"type": "integer", "description": "Max results (default 10)."},
            },
            "required": ["country_code"],
        },
    },
    {
        "name": "get_news_timeline",
        "description": (
            "Analyzes news volume over time for a topic (trend analysis). "
            "Timespan: 15min, 1h, 4h, 1d, 3d, 7d, 1m."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Topic to analyze."},
                "timespan": {"type": "string", "description": "Time period (default: 1d)."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_trending_topics",
        "description": "Identifies globally trending topics via GDELT with sentiment analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timespan": {"type": "string", "description": "Period: 1h, 4h, 1d, 3d, 7d (default: 1d)."},
                "tone_filter": {
                    "type": "string",
                    "description": "Sentiment filter: positive | negative | neutral. Empty = all.",
                },
            },
            "required": [],
        },
    },
]

SYSTEM_PROMPT = """You are a real-time information assistant specialised in:
1. Current weather anywhere in the world (Open-Meteo API, free, no API key)
2. Latest news (news-aggregator-mcp-server: RSS, HackerNews, GDELT)

Weather tools:
- resolve_city → converts a city name to coordinates
- get_current_weather → current weather by coordinates

News tools (choose the most appropriate for the context):
- get_news_by_category → news by category (tech/ai/general/business/crypto/science)
- search_rss_feeds → keyword search across all RSS feeds
- fetch_feed → fetch a specific RSS/Atom feed by URL
- get_hackernews_top → top stories from Hacker News (tech, startups)
- get_hackernews_trending → search HN stories by keywords
- search_global_news → global GDELT search (65+ languages, 100+ countries)
- get_news_by_country → news from a specific country (ISO-2 code)
- get_news_timeline → news volume over time for a topic
- get_trending_topics → globally trending topics

Instructions:
- Always call tools to get real-time data before responding.
- For weather questions: use resolve_city first if a city name is given, then get_current_weather.
- For general news: use get_news_by_category with the most relevant category.
- For tech/startup news: use get_hackernews_top.
- For international or country-specific news: use search_global_news or get_news_by_country.
- Present weather with clear units (°C, km/h, %).
- For news, present headlines with source and date.
- Cite data sources at the end of your response.
- If the question is not about weather or news, politely say so.
- Always respond in English."""


class WeatherNewsAgent:
    """Claude claude-opus-4-6 agent with 13 weather and news tools."""

    def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Dispatches a tool call and returns the result as a JSON string."""
        try:
            if tool_name == "get_current_weather":
                result = get_current_weather(**tool_input)
            elif tool_name == "resolve_city":
                city_info = resolve_location(tool_input.get("city_name", ""))
                if city_info:
                    result = {
                        "found": True,
                        "city": tool_input["city_name"],
                        "latitude": city_info["lat"],
                        "longitude": city_info["lon"],
                        "label": city_info["label"],
                    }
                else:
                    result = {
                        "found": False,
                        "city": tool_input.get("city_name", ""),
                        "message": (
                            "City not found. Use coordinates directly. "
                            "Default: Bogota (lat=4.71, lon=-74.07)."
                        ),
                    }
            elif tool_name in ALL_NEWS_TOOLS:
                result = ALL_NEWS_TOOLS[tool_name](**tool_input)
            else:
                result = {"error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            result = {"error": str(exc)}

        return json.dumps(result, ensure_ascii=False)

    def query(self, user_question: str) -> str:
        """Runs the agentic loop for a question. Returns the final answer."""
        messages: list[dict] = [{"role": "user", "content": user_question}]

        for _ in range(10):  # max iterations
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                return text_blocks[0].text if text_blocks else "(no response)"

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_str = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            # Unexpected stop_reason
            text_blocks = [b for b in response.content if b.type == "text"]
            return text_blocks[0].text if text_blocks else "(unexpected stop)"

        return "Agent reached the maximum number of iterations."

    def query_streaming(self, user_question: str) -> Iterator[str]:
        """Runs the agentic loop and yields the final text response."""
        messages: list[dict] = [{"role": "user", "content": user_question}]

        for _ in range(10):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        yield block.text
                return

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_str = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            for block in response.content:
                if block.type == "text":
                    yield block.text
            return
