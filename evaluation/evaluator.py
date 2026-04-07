"""
Evaluation module — measures the quality of the WeatherNewsAgent.

Metrics implemented
-------------------
1. Relevance Score    (0–10) : LLM-as-judge — does the answer address the question?
2. Keyword Coverage   (0–1)  : fraction of expected keywords found in the answer.
3. Groundedness Score (0–10) : LLM-as-judge — are the claims backed by real data?
4. Completeness Score (0–10) : LLM-as-judge — does the answer cover all aspects?
5. Pass / Fail               : composite score >= item's min_relevance_score threshold.

Composite formula:  0.40×Relevance + 0.20×(KeywordCov×10) + 0.20×Groundedness + 0.20×Completeness

The evaluation uses claude-haiku-4-5 (cheapest model) as the judge to keep costs low.
"""

import json
import re
import time
import logging
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import anthropic

# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────
EVAL_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"
JUDGE_MODEL       = "claude-haiku-4-5-20251001"
JUDGE_MAX_TOKENS  = 256
JUDGE_SLEEP_SEC   = 0.5   # pause between judge calls to avoid rate limiting
SCORE_MIN         = 0.0
SCORE_MAX         = 10.0

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    id: str
    category: str
    question: str
    answer: str
    relevance_score: float           # 0–10
    keyword_coverage: float          # 0–1
    groundedness_score: float        # 0–10
    completeness_score: float        # 0–10
    composite_score: float           # weighted average (0–10)
    min_required_score: float        # from dataset item
    passed: bool                     # composite_score >= min_required_score
    keywords_found: list[str]  = field(default_factory=list)
    keywords_missing: list[str] = field(default_factory=list)
    judge_feedback: str        = ""
    tool_calls_made: list[str] = field(default_factory=list)   # tools the agent actually called
    error: Optional[str]       = None


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator class
# ─────────────────────────────────────────────────────────────────────────────
class AgentEvaluator:
    """
    Runs the evaluation dataset against the WeatherNewsAgent and
    computes quality metrics for each question.
    """

    def __init__(self, api_key: str, agent):
        self.client  = anthropic.Anthropic(api_key=api_key)
        self.agent   = agent
        self.dataset = self._load_dataset()

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dataset() -> list[dict]:
        with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Metric: keyword coverage
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip accents for accent-insensitive matching."""
        return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode()

    def _keyword_coverage(self, answer: str, expected_keywords: list[str]) -> dict:
        """Returns fraction of expected keywords found in the answer (case- and accent-insensitive)."""
        if not expected_keywords:
            return {"coverage": 1.0, "found": [], "missing": []}
        answer_norm = self._normalize(answer)
        found   = [kw for kw in expected_keywords if self._normalize(kw) in answer_norm]
        missing = [kw for kw in expected_keywords if self._normalize(kw) not in answer_norm]
        return {
            "coverage": len(found) / len(expected_keywords),
            "found": found,
            "missing": missing,
        }

    # ------------------------------------------------------------------
    # Metric: LLM-as-judge score
    # ------------------------------------------------------------------

    def _judge_score(self, question: str, answer: str, criterion: str) -> tuple[float, str]:
        """
        Asks Claude Haiku to rate the answer on a named criterion (0–10).

        Args:
            question:  The original user question.
            answer:    The agent's response to evaluate.
            criterion: One of 'relevance', 'groundedness', 'completeness'.

        Returns:
            Tuple of (score clamped to 0–10, one-sentence feedback string).
        """
        criterion_prompts = {
            "relevance": (
                "Rate how relevant this answer is to the question on a scale of 0–10.\n"
                "Focus ONLY on whether the answer addresses the topic asked — not on whether you can personally verify the data.\n"
                "10 = directly and fully answers the question asked, 0 = completely ignores the question.\n"
                "Do NOT penalize answers for containing real-time data (weather readings, news headlines) that you cannot independently verify.\n"
            ),
            "groundedness": (
                "Rate how grounded this answer is in real data on a scale of 0–10.\n"
                "IMPORTANT CONTEXT: This is a real-time agent that calls live APIs (weather APIs, RSS feeds, HackerNews, GDELT). "
                "Treat all specific numbers, temperatures, headlines, and article details as accurate tool-fetched data — do NOT penalize for inability to verify them yourself.\n"
                "Penalize only obvious hallucinations: fabricated story titles, impossible values, or claims the agent made up without any tool basis.\n"
                "10 = answer clearly presents real data from live tools, 0 = answer invents fictional facts not grounded in any real source.\n"
            ),
            "completeness": (
                "Rate how completely the answer addresses all parts of the question on a scale of 0–10.\n"
                "Focus on whether all requested information is present — regardless of whether you can independently verify the data.\n"
                "10 = fully covers every aspect the question asked for, 0 = missing most important parts.\n"
            ),
        }
        instruction = criterion_prompts.get(criterion, criterion_prompts["relevance"])

        prompt = (
            f"{instruction}\n"
            f"Question: {question}\n\n"
            f"Answer:\n{answer}\n\n"
            "Respond with ONLY a valid JSON object — no extra text:\n"
            '{"score": <integer 0-10>, "feedback": "<one sentence explanation>"}'
        )

        for attempt in range(2):  # retry once on parse failure
            try:
                response = self.client.messages.create(
                    model=JUDGE_MODEL,
                    max_tokens=JUDGE_MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()

                # Extract the first JSON object from the response
                match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
                if not match:
                    log.warning("Judge returned no JSON on attempt %d: %s", attempt + 1, raw[:100])
                    continue

                data = json.loads(match.group())
                raw_score = float(data.get("score", SCORE_MAX / 2))
                # Clamp score to valid range
                score = max(SCORE_MIN, min(SCORE_MAX, raw_score))
                feedback = data.get("feedback", "")
                return score, feedback

            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                log.warning("Judge parse error on attempt %d: %s", attempt + 1, exc)
                time.sleep(JUDGE_SLEEP_SEC)
            except Exception as exc:
                log.error("Judge API error: %s", exc)
                return SCORE_MAX / 2, f"Judge error: {exc}"

        return SCORE_MAX / 2, "Could not parse judge response after retries"

    # ------------------------------------------------------------------
    # Tool call tracking
    # ------------------------------------------------------------------

    def _query_with_tool_tracking(self, question: str) -> tuple[str, list[str]]:
        """
        Runs the agent and tracks which tools were called.

        Returns:
            Tuple of (final answer text, list of tool names called in order).
        """
        import anthropic as _anthropic
        from mcp_servers.weather_tools import get_current_weather, resolve_location
        from mcp_servers.news_mcp_client import ALL_NEWS_TOOLS

        client = self.agent.client
        model  = self.agent.model
        messages: list[dict] = [{"role": "user", "content": question}]
        tools_called: list[str] = []

        # Import TOOLS and SYSTEM_PROMPT from orchestrator
        from agent.orchestrator import TOOLS, SYSTEM_PROMPT

        for _ in range(10):
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                text_blocks = [b for b in response.content if b.type == "text"]
                return (text_blocks[0].text if text_blocks else ""), tools_called

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tools_called.append(block.name)   # ← record tool name
                        result_str = self.agent._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                messages.append({"role": "user", "content": tool_results})
                continue

            text_blocks = [b for b in response.content if b.type == "text"]
            return (text_blocks[0].text if text_blocks else ""), tools_called

        return "Agent reached maximum iterations.", tools_called

    # ------------------------------------------------------------------
    # Single item evaluation
    # ------------------------------------------------------------------

    def evaluate_item(self, item: dict, verbose: bool = False) -> EvalResult:
        """
        Runs the agent on one evaluation item and computes all metrics.

        Args:
            item:    A single dataset entry dict.
            verbose: If True, prints progress to stdout.

        Returns:
            EvalResult with all metric scores and pass/fail.
        """
        question          = item["question"]
        min_score         = float(item.get("min_relevance_score", 7))

        # Run agent with tool tracking
        try:
            answer, tools_called = self._query_with_tool_tracking(question)
        except Exception as exc:
            log.error("Agent query failed for item %s: %s", item["id"], exc)
            return EvalResult(
                id=item["id"],
                category=item["category"],
                question=question,
                answer="",
                relevance_score=0.0,
                keyword_coverage=0.0,
                groundedness_score=0.0,
                completeness_score=0.0,
                composite_score=0.0,
                min_required_score=min_score,
                passed=False,
                error=str(exc),
            )

        if verbose:
            print(f"\n[{item['id']}] Q: {question[:80]}")
            print(f"  Tools called : {tools_called}")
            print(f"  A (snippet)  : {answer[:120]}...")

        # Keyword coverage (no API cost)
        kw = self._keyword_coverage(answer, item.get("expected_keywords", []))

        # LLM-as-judge metrics (3 API calls)
        relevance,    rel_feedback = self._judge_score(question, answer, "relevance")
        time.sleep(JUDGE_SLEEP_SEC)
        groundedness, _            = self._judge_score(question, answer, "groundedness")
        time.sleep(JUDGE_SLEEP_SEC)
        completeness, _            = self._judge_score(question, answer, "completeness")

        # Composite score: 40% relevance, 20% keyword coverage, 20% groundedness, 20% completeness
        composite = round(
            relevance    * 0.40
            + kw["coverage"] * 10 * 0.20
            + groundedness   * 0.20
            + completeness   * 0.20,
            2,
        )

        passed = composite >= min_score

        return EvalResult(
            id=item["id"],
            category=item["category"],
            question=question,
            answer=answer,
            relevance_score=relevance,
            keyword_coverage=kw["coverage"],
            groundedness_score=groundedness,
            completeness_score=completeness,
            composite_score=composite,
            min_required_score=min_score,
            passed=passed,
            keywords_found=kw["found"],
            keywords_missing=kw["missing"],
            judge_feedback=rel_feedback,
            tool_calls_made=tools_called,
        )

    # ------------------------------------------------------------------
    # Full evaluation run
    # ------------------------------------------------------------------

    def run_evaluation(
        self,
        ids: Optional[list[str]] = None,
        verbose: bool = True,
    ) -> dict:
        """
        Runs the full (or filtered) evaluation dataset and returns aggregated results.

        Args:
            ids:     Optional list of item IDs to evaluate (default: all).
            verbose: If True, prints progress to stdout.

        Returns:
            Dict with 'results' list and 'summary' metrics dict.
        """
        items = self.dataset
        if ids:
            items = [it for it in items if it["id"] in ids]

        results: list[EvalResult] = []
        for item in items:
            if verbose:
                print(f"Evaluating {item['id']} ({item['category']})...")
            result = self.evaluate_item(item, verbose=verbose)
            results.append(result)

        # Aggregate metrics
        successful = [r for r in results if r.error is None]
        n = max(len(successful), 1)

        by_category: dict[str, list[float]] = {}
        for r in successful:
            by_category.setdefault(r.category, []).append(r.composite_score)

        category_avgs = {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in by_category.items()
        }

        passed_count = sum(1 for r in successful if r.passed)

        summary = {
            "total_questions":      len(results),
            "successful":           len(successful),
            "failed":               len(results) - len(successful),
            "passed_threshold":     passed_count,
            "pass_rate":            round(passed_count / n, 2),
            "avg_relevance":        round(sum(r.relevance_score    for r in successful) / n, 2),
            "avg_keyword_coverage": round(sum(r.keyword_coverage    for r in successful) / n, 3),
            "avg_groundedness":     round(sum(r.groundedness_score  for r in successful) / n, 2),
            "avg_completeness":     round(sum(r.completeness_score  for r in successful) / n, 2),
            "avg_composite_score":  round(sum(r.composite_score     for r in successful) / n, 2),
            "by_category":          category_avgs,
        }

        return {
            "results": [asdict(r) for r in results],
            "summary": summary,
        }
