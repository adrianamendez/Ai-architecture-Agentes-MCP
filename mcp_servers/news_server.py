"""
News MCP Server — uses the official news-aggregator-mcp-server package.

Installation:
    pip install news-aggregator-mcp-server

To run as a standalone MCP server (stdio transport):
    news-aggregator-server          # CLI command provided by the package
    # or:
    python -m src.server            # package module

Tools exposed (11 total, no API key required):
  RSS/Atom   : fetch_feed, get_news_by_category, search_rss_feeds, list_feed_catalog
  HackerNews : get_hackernews_top, get_hackernews_story, get_hackernews_trending
  GDELT      : search_global_news, get_news_timeline, get_news_by_country, get_trending_topics

Data sources:
  - 19 predefined RSS feeds in 6 categories (tech, ai, general, business, crypto, science)
  - HackerNews API (https://hacker-news.firebaseio.com)
  - GDELT Project (https://gdeltproject.org) — 65+ languages, 100+ countries

Integration with Claude Agent SDK:
    options = ClaudeAgentOptions(
        mcp_servers={
            "news": {"command": "news-aggregator-server"}
        }
    )
"""

# The real MCP server is provided by the installed package.
# This file serves as a reference and documentation entry point.

# For use from Python code (synchronous bridge):
from mcp_servers.news_mcp_client import ALL_NEWS_TOOLS  # noqa: F401

if __name__ == "__main__":
    # Start the official MCP server from the package
    from src.server import main
    main()
