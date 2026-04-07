"""
Weather & News Assistant — Streamlit Application
=================================================
Architecture
------------
                ┌─────────────────────────────────────┐
                │         Streamlit Frontend           │
                │   Tab Chat | Tab Eval | Tab About    │
                └────────────────┬────────────────────┘
                                 │ user question
                                 ▼
                ┌─────────────────────────────────────┐
                │       WeatherNewsAgent               │
                │  Claude claude-opus-4-6 + Tool Use   │
                │  (agentic loop: plan→call→respond)   │
                └────────┬──────────────┬─────────────┘
                         │              │
              ┌──────────▼──┐    ┌──────▼────────────────────┐
              │ Weather MCP │    │ news-aggregator-mcp-server │
              │  (custom)   │    │       (pip install)        │
              │ Open-Meteo  │    │ RSS · HackerNews · GDELT   │
              └─────────────┘    └────────────────────────────┘

Data sources (all free, no API key required):
  • Weather → Open-Meteo  https://open-meteo.com
  • News    → news-aggregator-mcp-server (RSS feeds, HackerNews, GDELT)

Evaluation:
  • 4 quantitative metrics (LLM-as-judge + keyword matching)
  • 10 questions in evaluation/eval_dataset.json
  • Radar and bar charts in the Evaluation tab

Run:
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library imports
# ─────────────────────────────────────────────────────────────────────────────
import json           # result serialisation
import os             # environment variables (API key)
import time           # pauses between judge calls
from pathlib import Path  # filesystem paths

# ─────────────────────────────────────────────────────────────────────────────
# Third-party dependencies
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd               # result tables
import plotly.graph_objects as go  # radar and bar charts
import streamlit as st            # UI framework
from dotenv import load_dotenv    # load .env in local development

# Load environment variables from .env if present
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config — must be the first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather & News Assistant",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation (persists across Streamlit reruns)
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    # Conversation history: list of dicts {"role": "user"|"assistant", "content": str}
    st.session_state.messages = []

if "agent" not in st.session_state:
    # Agent instance (created once per session)
    st.session_state.agent = None

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = False

if "api_key" not in st.session_state:
    # Never pre-filled from environment in the UI — each user must enter their own key.
    # The env var is only read for local development convenience (not in production).
    st.session_state.api_key = ""

if "eval_results" not in st.session_state:
    # Results from the last evaluation run
    st.session_state.eval_results = None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Configuration and quick presets
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")

    # ── API Key ──────────────────────────────────────────────────────────────
    api_key_input = st.text_input(
        "Anthropic API Key",
        key="api_key_widget",      # explicit key lets us reset the widget on clear
        type="password",
        placeholder="sk-ant-...",
        help=(
            "Your Anthropic API key. "
            "Stored only in your browser session — never saved, never shared with other users. "
            "Cleared automatically when you close the tab."
        ),
    )

    if api_key_input:
        if not api_key_input.startswith("sk-ant-"):
            st.error("Invalid key format. Anthropic API keys start with 'sk-ant-'.")
        elif api_key_input != st.session_state.api_key:
            # Key changed — invalidate existing agent so it is rebuilt with new key
            st.session_state.api_key = api_key_input
            st.session_state.api_key_set = True
            st.session_state.agent = None

    col_status, col_clear = st.columns([3, 1])
    with col_status:
        if st.session_state.api_key_set:
            masked = "sk-ant-..." + st.session_state.api_key[-4:]
            st.success(f"✅ Key set ({masked})")
        else:
            st.warning("⚠️ Enter your Anthropic API key")
    with col_clear:
        if st.session_state.api_key_set and st.button("🗑️", help="Clear API key"):
            st.session_state.api_key = ""
            st.session_state.api_key_set = False
            st.session_state.agent = None
            st.session_state.api_key_widget = ""   # resets the text input widget itself
            st.rerun()

    st.markdown("---")

    # ── Weather presets ───────────────────────────────────────────────────────
    st.subheader("🌍 Quick weather")
    presets = {
        "Bogotá, Colombia": (4.71, -74.07),
        "New York, USA": (40.71, -74.01),
        "London, UK": (51.51, -0.13),
        "Tokyo, Japan": (35.68, 139.69),
        "Sydney, Australia": (-33.87, 151.21),
        "Paris, France": (48.85, 2.35),
    }
    selected_preset = st.selectbox("City", list(presets.keys()))
    if st.button("🌡️ Get weather"):
        # Inject question into history → processed on the next rerun
        st.session_state.messages.append({
            "role": "user",
            "content": f"What is the current weather in {selected_preset}?",
        })
        st.rerun()

    st.markdown("---")

    # ── News presets ──────────────────────────────────────────────────────────
    st.subheader("📰 Quick news")
    topic_options = ["Top headlines", "Technology", "AI", "Business", "Crypto", "Science", "Climate"]
    selected_topic = st.selectbox("Topic", topic_options)
    if st.button("📰 Get news"):
        prompt = (
            "What are the latest top news headlines?"
            if selected_topic == "Top headlines"
            else f"What are the latest news about {selected_topic}?"
        )
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    # ── Credits ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Powered by **Claude claude-opus-4-6** · Open-Meteo · "
        "news-aggregator-mcp-server"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — lazy-load the agent (instantiated once per session)
# ─────────────────────────────────────────────────────────────────────────────
def _sanitize_error(exc: Exception) -> str:
    """Returns a safe error message with any API key substring redacted."""
    msg = str(exc)
    key = st.session_state.get("api_key", "")
    if key and key in msg:
        msg = msg.replace(key, "sk-ant-***REDACTED***")
    return msg


def get_agent():
    """Returns (or creates) the WeatherNewsAgent instance for this session."""
    if st.session_state.agent is None:
        api_key = st.session_state.api_key
        if not api_key:
            st.error("❌ API key not configured. Enter it in the sidebar.")
            st.stop()
        from agent.orchestrator import WeatherNewsAgent
        st.session_state.agent = WeatherNewsAgent(api_key=api_key)
    return st.session_state.agent


# ─────────────────────────────────────────────────────────────────────────────
# Main layout — 3 tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_chat, tab_eval, tab_about = st.tabs(["💬 Chat", "📊 Evaluation", "ℹ️ About"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.title("🌤️ Weather & News Assistant")
    st.markdown(
        "Ask me about **current weather** or **latest news**. "
        "I use real-time data from Open-Meteo and multiple RSS/HN/GDELT sources."
    )

    # ── Display conversation history ──────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Process pending message (injected by sidebar buttons) ─────────────────
    needs_reply = (
        st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
        and (
            len(st.session_state.messages) < 2
            or st.session_state.messages[-2]["role"] != "assistant"
        )
    )
    if needs_reply and st.session_state.api_key_set:
        last_user_msg = st.session_state.messages[-1]["content"]
        agent = get_agent()
        with st.chat_message("assistant"):
            with st.spinner("Fetching real-time data…"):
                try:
                    answer = agent.query(last_user_msg)
                except Exception as exc:
                    answer = f"❌ Error: {_sanitize_error(exc)}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # ── Chat input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about weather or news…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.api_key_set:
            st.error("Please configure your Anthropic API key in the sidebar.")
        else:
            agent = get_agent()
            with st.chat_message("assistant"):
                with st.spinner("Fetching real-time data…"):
                    try:
                        answer = agent.query(prompt)
                    except Exception as exc:
                        answer = f"❌ Error: {exc}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.title("📊 Agent Evaluation Panel")
    st.markdown(
        """
        This panel evaluates the quality of agent responses using **4 metrics**:

        | Metric | Weight | Method | Scale |
        |--------|--------|--------|-------|
        | **Relevance** | 40% | LLM-as-judge (claude-haiku-4-5-20251001) | 0–10 |
        | **Keyword Coverage** | 20% | String matching (accent-insensitive) | 0–1 |
        | **Groundedness** | 20% | LLM-as-judge | 0–10 |
        | **Completeness** | 20% | LLM-as-judge | 0–10 |

        The **composite score** is the weighted average (scale 0–10).
        The judge model is **claude-haiku-4-5-20251001** (fast and cost-efficient for evaluation).
        """
    )

    # ── Evaluation dataset ────────────────────────────────────────────────────
    eval_dataset_path = Path("evaluation/eval_dataset.json")
    if eval_dataset_path.exists():
        with open(eval_dataset_path) as f:
            eval_items = json.load(f)

        st.subheader("📋 Evaluation Dataset (10 questions)")
        df_dataset = pd.DataFrame([
            {
                "ID": it["id"],
                "Category": it["category"],
                "Question": it["question"],
                "Expected Keywords": ", ".join(it.get("expected_keywords", [])),
                "Expected Tools": ", ".join(it.get("expected_tools", [])),
            }
            for it in eval_items
        ])
        st.dataframe(df_dataset, use_container_width=True, hide_index=True)

    # ── Run evaluation buttons ────────────────────────────────────────────────
    col_run, col_quick = st.columns(2)
    with col_run:
        run_full = st.button(
            "▶️ Full evaluation (10 questions)",
            disabled=not st.session_state.api_key_set,
            type="primary",
        )
    with col_quick:
        run_quick = st.button(
            "⚡ Quick evaluation (3 questions: w1, n1, m1)",
            disabled=not st.session_state.api_key_set,
        )

    if not st.session_state.api_key_set:
        st.info("ℹ️ Configure your Anthropic API key in the sidebar to run the evaluation.")

    # ── Execute evaluation ─────────────────────────────────────────────────────
    if run_full or run_quick:
        from evaluation.evaluator import AgentEvaluator

        agent = get_agent()
        evaluator = AgentEvaluator(
            api_key=st.session_state.api_key,
            agent=agent,
        )

        # Filter to quick subset if requested
        selected_ids = ["w1", "n1", "m1"] if run_quick else None
        items_to_eval = evaluator.dataset
        if selected_ids:
            items_to_eval = [it for it in items_to_eval if it["id"] in selected_ids]

        # Progress bar
        progress_bar = st.progress(0, text="Starting evaluation…")
        status_text = st.empty()

        results_list = []
        for i, item in enumerate(items_to_eval):
            status_text.text(f"Evaluating [{item['id']}]: {item['question'][:60]}…")
            result = evaluator.evaluate_item(item)
            results_list.append(result)
            progress_bar.progress(
                (i + 1) / len(items_to_eval),
                text=f"Question {i+1}/{len(items_to_eval)} completed",
            )
            time.sleep(0.1)  # small pause to update the UI

        # ── Aggregate metrics ─────────────────────────────────────────────────
        from dataclasses import asdict

        successful = [r for r in results_list if r.error is None]

        # Group composite scores by category
        by_category: dict[str, list[float]] = {}
        for r in successful:
            by_category.setdefault(r.category, []).append(r.composite_score)

        # Average per category
        category_avgs = {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in by_category.items()
        }

        # Global metrics
        n = max(len(successful), 1)
        passed_count = sum(1 for r in successful if r.passed)
        summary = {
            "total_questions": len(results_list),
            "successful": len(successful),
            "failed": len(results_list) - len(successful),
            "passed_threshold": passed_count,
            "pass_rate": round(passed_count / n, 2),
            "avg_relevance": round(sum(r.relevance_score for r in successful) / n, 2),
            "avg_keyword_coverage": round(sum(r.keyword_coverage for r in successful) / n, 3),
            "avg_groundedness": round(sum(r.groundedness_score for r in successful) / n, 2),
            "avg_completeness": round(sum(r.completeness_score for r in successful) / n, 2),
            "avg_composite_score": round(sum(r.composite_score for r in successful) / n, 2),
            "by_category": category_avgs,
        }

        # Persist in session so results survive a rerun
        st.session_state.eval_results = {
            "results": [asdict(r) for r in results_list],
            "summary": summary,
        }

        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Evaluation complete: {len(results_list)} questions processed")

    # ── Display results ────────────────────────────────────────────────────────
    if st.session_state.eval_results:
        ev = st.session_state.eval_results
        summary = ev["summary"]
        results = ev["results"]

        st.markdown("---")
        st.subheader("📈 Global Metrics")

        # Metric cards
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Relevance", f"{summary['avg_relevance']}/10",
                  help="LLM-as-judge: does the answer address the question?")
        m2.metric("Keywords", f"{summary['avg_keyword_coverage']:.0%}",
                  help="Fraction of expected keywords found in the answer")
        m3.metric("Groundedness", f"{summary['avg_groundedness']}/10",
                  help="LLM-as-judge: is the data supported by tool results?")
        m4.metric("Completeness", f"{summary['avg_completeness']}/10",
                  help="LLM-as-judge: does the answer cover all aspects?")
        m5.metric("Composite Score", f"{summary['avg_composite_score']}/10",
                  help="Weighted average 40/20/20/20")
        m6.metric(
            "Pass Rate",
            f"{summary.get('pass_rate', 0):.0%}",
            delta=f"{summary.get('passed_threshold', 0)}/{summary['successful']} passed",
            help="Fraction of questions that met their minimum composite score threshold",
        )

        # ── Radar chart (holistic quality view) ───────────────────────────────
        st.subheader("🕸️ Quality Radar")
        radar_categories = ["Relevance", "Keywords×10", "Groundedness", "Completeness"]
        radar_values = [
            summary["avg_relevance"],
            summary["avg_keyword_coverage"] * 10,  # scale to 0–10
            summary["avg_groundedness"],
            summary["avg_completeness"],
        ]
        fig_radar = go.Figure(
            go.Scatterpolar(
                r=radar_values + [radar_values[0]],        # close the polygon
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name="Agent scores",
                line_color="#4F8BF9",
                fillcolor="rgba(79,139,249,0.25)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=False,
            height=380,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Bar chart by category ──────────────────────────────────────────────
        if summary.get("by_category"):
            st.subheader("📊 Composite Score by Category")
            cats = list(summary["by_category"].keys())
            scores = list(summary["by_category"].values())
            fig_bar = go.Figure(
                go.Bar(
                    x=cats,
                    y=scores,
                    marker_color=["#4F8BF9", "#F97B4F", "#4FF98B", "#F9D94F", "#AF4FF9"],
                    text=[f"{s}/10" for s in scores],
                    textposition="outside",
                )
            )
            fig_bar.update_layout(
                yaxis=dict(range=[0, 11], title="Composite Score (0–10)"),
                xaxis_title="Category",
                height=350,
                margin=dict(t=30, b=40),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Detailed table per question ────────────────────────────────────────
        st.subheader("🔍 Results by Question")
        df_results = pd.DataFrame([
            {
                "ID": r["id"],
                "Category": r["category"],
                "Question": r["question"][:55] + ("…" if len(r["question"]) > 55 else ""),
                "Relevance": r["relevance_score"],
                "Keywords": f"{r['keyword_coverage']:.0%}",
                "Groundedness": r["groundedness_score"],
                "Completeness": r["completeness_score"],
                "Composite": r["composite_score"],
                "Threshold": r.get("min_required_score", "—"),
                "Pass": "✅" if r.get("passed") else ("❌" if not r.get("error") else "⚠️"),
            }
            for r in results
        ])
        st.dataframe(
            df_results.style.background_gradient(
                subset=["Relevance", "Groundedness", "Completeness", "Composite"],
                cmap="RdYlGn",
                vmin=0,
                vmax=10,
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ── Full agent answers (expandable) ────────────────────────────────────
        st.subheader("📝 Full Agent Answers")
        for r in results:
            if r.get("error"):
                icon = "⚠️"
                score_label = "[error]"
            elif r.get("passed"):
                icon = "✅"
                score_label = f"[{r['composite_score']}/10 ≥ {r.get('min_required_score', '?')}]"
            else:
                icon = "❌"
                score_label = f"[{r['composite_score']}/10 < {r.get('min_required_score', '?')}]"

            with st.expander(f"{icon} {score_label} [{r['id']}] {r['question'][:65]}"):
                if r.get("error"):
                    st.error(f"Error: {r['error']}")
                else:
                    st.markdown(r["answer"])
                    st.divider()
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.caption(
                            f"**Keywords found:** {', '.join(r['keywords_found']) or 'none'}  \n"
                            f"**Keywords missing:** {', '.join(r['keywords_missing']) or 'none'}"
                        )
                    with col_b:
                        tools_called = r.get("tool_calls_made") or []
                        st.caption(
                            f"**Tools called:** {', '.join(tools_called) or 'none'}"
                        )
                    with col_c:
                        if r.get("judge_feedback"):
                            st.info(f"💬 Judge: {r['judge_feedback']}")

        # ── Download results ───────────────────────────────────────────────────
        st.download_button(
            label="⬇️ Download results (JSON)",
            data=json.dumps(ev, indent=2, ensure_ascii=False),
            file_name="eval_results.json",
            mime="application/json",
        )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT / DOCUMENTATION
# ═════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.title("ℹ️ Architecture & Documentation")

    st.markdown(
        """
## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend (app.py)                  │
│   ┌────────────┐   ┌─────────────────────┐   ┌───────────────┐  │
│   │  Tab Chat  │   │   Tab Evaluation    │   │  Tab About    │  │
│   └─────┬──────┘   └──────────┬──────────┘   └───────────────┘  │
└─────────┼───────────────────────────────────────────────────────┘
          │ user question
          ▼
┌──────────────────────────────────────────────────────────────────┐
│                 WeatherNewsAgent (agent/orchestrator.py)         │
│                                                                  │
│  • Model: Claude claude-opus-4-6 (Anthropic API)                 │
│  • Agentic loop: while stop_reason == "tool_use": ...           │
│  • 13 registered tools (2 weather + 11 news)                    │
└────────┬──────────────────────────────────┬─────────────────────┘
         │                                  │
         ▼                                  ▼
┌────────────────────┐    ┌─────────────────────────────────────────┐
│  Weather MCP Server│    │    news-aggregator-mcp-server (pip)     │
│  (FastMCP, custom) │    │                                         │
│                    │    │  RSS/Atom (4 tools):                    │
│  Open-Meteo API    │    │    • get_news_by_category               │
│  lat=4.71,lon=-74  │    │      (tech/ai/general/business/crypto/  │
│  No API key ✓      │    │       science — 19 predefined feeds)    │
│                    │    │    • search_rss_feeds                   │
│  Tools:            │    │    • fetch_feed (any RSS URL)           │
│  • resolve_city    │    │    • list_feed_catalog                  │
│  • get_weather     │    │                                         │
│                    │    │  HackerNews (3 tools):                  │
│                    │    │    • get_hackernews_top                 │
│                    │    │    • get_hackernews_trending            │
│                    │    │    • get_hackernews_story               │
│                    │    │                                         │
│                    │    │  GDELT Global News (4 tools):           │
│                    │    │    • search_global_news (65+ languages) │
│                    │    │    • get_news_by_country (100+ countries│
│                    │    │    • get_news_timeline                  │
│                    │    │    • get_trending_topics                │
└────────────────────┘    └─────────────────────────────────────────┘
```

## Project Modules

| File | Responsibility |
|------|----------------|
| `app.py` | Streamlit UI: chat, evaluation, documentation |
| `agent/orchestrator.py` | Claude agent with tool-use loop and 13 tools |
| `mcp_servers/weather_tools.py` | Synchronous functions → Open-Meteo REST API |
| `mcp_servers/weather_server.py` | Standalone FastMCP server (stdio transport) |
| `mcp_servers/news_mcp_client.py` | Synchronous bridge to `news-aggregator-mcp-server` |
| `mcp_servers/news_server.py` | Reference file + news server entry point |
| `evaluation/evaluator.py` | LLM-as-judge metrics + keyword coverage |
| `evaluation/eval_dataset.json` | 10 evaluation questions (weather, news, mixed, edge) |

## Data Sources (all free, no API key required)

| Source | Type | Coverage |
|--------|------|----------|
| **Open-Meteo** | REST API | Global weather by coordinates |
| **RSS/Atom feeds** | RSS | 19 sources (TechCrunch, BBC, Reuters, Bloomberg…) |
| **Hacker News** | Public REST API | Tech, startups, community |
| **GDELT Project** | REST API | 65+ languages, 100+ countries, trend analysis |

## Evaluation Framework

```
Dataset question
       │
       ▼
  WeatherNewsAgent.query()
       │
       ▼
  Agent answer
       │
       ├─── Keyword Coverage ──────► fraction of expected keywords found (0–1)
       │
       ├─── Relevance Judge ───────► claude-haiku-4-5-20251001 → score 0–10
       │
       ├─── Groundedness Judge ────► claude-haiku-4-5-20251001 → score 0–10
       │
       └─── Completeness Judge ────► claude-haiku-4-5-20251001 → score 0–10
                   │
                   ▼
        Composite = 0.40×Rel + 0.20×KW×10 + 0.20×Grd + 0.20×Cmp
                   │
                   ▼
        Pass/Fail: Composite ≥ min_relevance_score (per dataset item)
```

## Standalone News MCP Server

The `news-aggregator-mcp-server` package includes a standard MCP server
that can run as an independent process:

```bash
# Install
pip install news-aggregator-mcp-server

# Run server (stdio transport — compatible with any MCP client)
news-aggregator-server

# Integrate with Claude Agent SDK
options = ClaudeAgentOptions(
    mcp_servers={"news": {"command": "news-aggregator-server"}}
)
```

## Full Setup

```bash
# 1. Clone / unzip the project
cd mcp_agents_epam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...

# 4. Run the application
streamlit run app.py
```
        """
    )
