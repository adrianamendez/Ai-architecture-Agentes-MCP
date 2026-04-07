# Practical Task: Agents, MCP

A production-style AI agent that answers natural-language questions about **real-time weather** and **current news** using the Anthropic Claude API and the Model Context Protocol (MCP) pattern.

---

## Task Overview

This project demonstrates how to build an LLM-powered agent that autonomously selects and invokes external tools to answer user questions. It combines a custom MCP weather server with an off-the-shelf news aggregation MCP server, wired together through an agentic tool-use loop driven by Claude claude-opus-4-6.

### What this project demonstrates

| Layer | What is shown |
|-------|---------------|
| **MCP Server 1 — Weather** | Custom FastMCP server wrapping the free Open-Meteo REST API; city geocoding via built-in dictionary + Open-Meteo geocoding |
| **MCP Server 2 — News** | Integration of the `news-aggregator-mcp-server` pip package (RSS/Atom, HackerNews Firebase API, GDELT global news) |
| **Agent Orchestrator** | Agentic tool-use loop with Claude claude-opus-4-6: `while stop_reason == "tool_use"` → execute tool → append result → re-prompt |
| **Streamlit UI** | Conversational chat interface + evaluation panel with radar/bar charts |
| **Evaluation Framework** | LLM-as-judge (`claude-haiku-4-5-20251001`) scoring 4 metrics; accent-insensitive keyword coverage; composite pass/fail per question |

---

## Architecture

```
┌─────────────┐   questions   ┌──────────────────────┐
│  Streamlit  │ ────────────► │  WeatherNewsAgent    │
│   app.py    │ ◄──────────── │  (orchestrator.py)   │
└─────────────┘   answers     └──────────┬───────────┘
                                         │ tool_use loop
                          ┌──────────────┼──────────────┐
                          ▼              ▼              ▼
                   ┌────────────┐ ┌──────────┐ ┌──────────────┐
                   │ Open-Meteo │ │   RSS    │ │ HackerNews / │
                   │  (custom   │ │  feeds   │ │   GDELT      │
                   │ MCP server)│ │(pip MCP) │ │  (pip MCP)   │
                   └────────────┘ └──────────┘ └──────────────┘
```

**Agentic loop flow:**
```
User question
      │
      ▼
claude-opus-4-6  ──tool_use──►  _execute_tool(name, input)
      ▲                              │
      │◄──── tool_result ────────────┘
      │                          (repeat up to 10 iterations)
      └──end_turn──► Final answer returned to user
```

---

## Methodology

The project follows the **MCP (Model Context Protocol)** pattern: tools are defined with structured JSON schemas and registered with the Claude API. The model decides autonomously which tool to call, with what arguments, and when to stop.

**Evaluation methodology:** Each agent response is scored by a second model (`claude-haiku-4-5-20251001`) on three criteria — relevance, groundedness, and completeness — plus accent-insensitive keyword-coverage matching. The judge is instructed to treat real-time tool-fetched data as accurate. A weighted composite score determines pass/fail against a per-question threshold.

```
Composite = 0.40 × Relevance + 0.20 × (Keyword Coverage × 10) + 0.20 × Groundedness + 0.20 × Completeness
Pass/Fail: Composite ≥ min_relevance_score (configured per evaluation item)
```

---

## Project Structure

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI: chat, evaluation panel, documentation |
| `agent/orchestrator.py` | Claude agent with tool-use loop and 13 tools |
| `mcp_servers/weather_tools.py` | Synchronous functions → Open-Meteo REST API |
| `mcp_servers/weather_server.py` | Standalone FastMCP server (stdio transport) |
| `mcp_servers/news_mcp_client.py` | Synchronous bridge to `news-aggregator-mcp-server` |
| `mcp_servers/news_server.py` | Reference file + news server entry point |
| `evaluation/evaluator.py` | LLM-as-judge metrics + keyword coverage + pass/fail |
| `evaluation/eval_dataset.json` | 10 evaluation questions (weather, news, mixed, edge) |
| `notebook.ipynb` | Full pipeline walkthrough notebook |

---

## Data and Resources Used

| Source | Type | Coverage | API Key |
|--------|------|----------|---------|
| **Open-Meteo** | REST API | Global weather by coordinates | Not required |
| **RSS/Atom feeds** | RSS | 19 sources: TechCrunch, Wired, BBC, Reuters, Bloomberg, CoinDesk, Nature, IEEE Spectrum, MIT Tech Review, FT, CNBC… | Not required |
| **Hacker News** | Firebase REST API | Tech/startup community stories (top, new, best, ask, show, jobs) | Not required |
| **GDELT Project** | REST API | 65+ languages, 100+ countries, trend analysis | Not required |

> Only the **Anthropic API key** is required to run the project.

---

## Installation

```bash
# 1. Clone or unzip the project
cd mcp_agents_epam

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Configure your Anthropic API key
cp .env.example .env
# Edit .env and set:  ANTHROPIC_API_KEY=sk-ant-...

# 4a. Run the Streamlit app
streamlit run app.py

# 4b. Or open the notebook
jupyter notebook notebook.ipynb
```

---

## References

| Resource | URL |
|----------|-----|
| Anthropic Claude API docs | https://docs.anthropic.com |
| Model Context Protocol spec | https://modelcontextprotocol.io |
| Anthropic Tool Use guide | https://docs.anthropic.com/en/docs/build-with-claude/tool-use |
| Open-Meteo (weather API) | https://open-meteo.com |
| news-aggregator-mcp-server (PyPI) | https://pypi.org/project/news-aggregator-mcp-server |
| HackerNews Firebase API | https://hacker-news.firebaseio.com |
| GDELT Project | https://gdeltproject.org |

---

## Other Notes

- The `news-aggregator-mcp-server` package uses German field names internally (`titel`, `zusammenfassung`, `punkte`, etc.). The `news_mcp_client.py` wrapper normalises all fields to English before returning data, so both the agent and notebook receive consistent English-keyed responses.
- GDELT may enforce rate limits or time out. The code catches `TimeoutError` and returns a structured error dict, which the retry logic handles with exponential backoff (5 s → 10 s → 20 s).
- The async internals of the news package are bridged to synchronous calls via a `_run()` helper that uses a `ThreadPoolExecutor` fallback when a running event loop is detected (required for Streamlit and Jupyter compatibility).
- Running the full evaluation suite (~40 API calls) costs approximately **$0.05–0.15 USD** with `claude-opus-4-6` (agent) + `claude-haiku-4-5-20251001` (judge).

---

## Summary & Key Learnings

### What was implemented and demonstrated

| Component | Implementation |
|-----------|----------------|
| **Custom MCP Weather Server** | `FastMCP` server exposing `resolve_city` and `get_current_weather`; wraps Open-Meteo REST API with geocoding |
| **News MCP Integration** | Synchronous bridge to `news-aggregator-mcp-server` covering RSS (19 feeds), HackerNews, and GDELT |
| **Agentic Tool-Use Loop** | Claude claude-opus-4-6 drives a `while stop_reason == "tool_use"` loop up to 10 iterations; all 13 tools registered via JSON schema |
| **Streamlit UI** | Chat tab with conversation history + Evaluation tab with metric cards, radar chart, bar chart, and per-question expanders |
| **LLM-as-Judge Evaluation** | 4-metric framework (relevance, keyword coverage, groundedness, completeness); `claude-haiku-4-5-20251001` as judge with real-time-data-aware prompts; pass/fail per question with configurable threshold |
| **Tool-call Tracking** | Evaluator records which tools were actually called per question for debugging and analysis |

### Why this matters

- **MCP is the emerging standard** for connecting AI models to external data sources. Building a custom MCP server from scratch shows how any REST API can be wrapped and exposed to an LLM as a callable tool.
- **Agentic loops require careful design.** The orchestrator must handle multi-step reasoning (e.g., geocode a city → fetch weather), error recovery, and a maximum-iteration safeguard to avoid runaway costs.
- **Evaluation is non-trivial for open-ended answers.** Using another LLM as a judge (LLM-as-judge pattern) enables scalable quality measurement without hand-labelling every response — a standard technique in production AI systems.
- **Data source diversity matters.** Combining RSS feeds, HackerNews, and GDELT gives the agent broad coverage (tech, community, global/multi-language) that no single source could provide.
- **Async-to-sync bridging** is a real engineering challenge when integrating async libraries into synchronous frameworks like Streamlit — solved here with a `ThreadPoolExecutor` fallback.

### Next steps

- **Streaming responses** — use the Anthropic streaming API so users see partial answers as they arrive rather than waiting for the full response.
- **Persistent conversation memory** — store tool results across turns so the agent can reference earlier answers without re-fetching.
- **More MCP servers** — add finance data (Yahoo Finance), weather forecasts (multi-day), or a web-search tool.
- **Automated evaluation CI** — run the evaluation suite on every code change and track metric trends over time.
- **Tool-use optimisation** — analyse `tool_calls_made` logs to identify which tools are under- or over-used and fine-tune the system prompt accordingly.
- **Multi-agent architecture** — delegate specialised sub-tasks (weather, news, summarisation) to separate agents orchestrated by a routing layer.
