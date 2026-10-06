#!/usr/bin/env python3
"""
UmbrelaJudge: LLM-based RAG judge using the UMBRELA relevance-grading prompts.

UMBRELA (https://github.com/castorini/umbrela) is a faithful, open-source
reproduction of Microsoft Bing's "DNA" LLM-based relevance judge from
Thomas et al., "Large Language Models can Accurately Predict Searcher
Preferences" (https://arxiv.org/abs/2309.10621), originally used to produce
query-passage relevance judgments (qrels) that closely match human TREC
assessors.

This judge reuses UMBRELA's exact prompt templates (copied verbatim into
`prompt_templates/`) but applies them to the RAG setting: for each
(topic, run) pair, the full RAG response is treated as the "passage" being
graded for relevance to the topic's "query" (title), on UMBRELA's standard
0-3 scale:

    0 = the response has nothing to do with the query
    1 = the response seems related to the query but does not answer it
    2 = the response has some answer for the query, but it may be unclear or
        hidden amongst extraneous information
    3 = the response is dedicated to the query and contains the exact answer

Two prompt styles are supported, matching UMBRELA's own `prompt_type`
options:

    "bing"  (default) - the original step-by-step Bing/DNA prompt that asks
            the LLM to reason about search intent, match, and trustworthiness
            before giving a final score. Supports an optional few-shot
            variant with UMBRELA's own worked examples baked in.
    "basic" - UMBRELA's simpler prompt without explicit step-by-step
            reasoning instructions.

`judge()` grades every response directly and reports the UMBRELA_RELEVANCE
(mean graded score) and UMBRELA_RELEVANT (fraction graded >=2, the standard
graded-to-binary relevance threshold) measures. `create_qrels()` optionally
runs the same grading to produce standard Qrels (topic_id, doc_id=run_id,
grade), for workflows that want UMBRELA-style relevance judgments as a
first-class, reusable artifact (UMBRELA's original purpose) rather than only
a leaderboard.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    QrelRow,
    Report,
    Request,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

from .prompts import render_prompt

GRADE_MAX = 3  # UMBRELA's graded relevance scale: 0 (not relevant) to 3 (fully relevant)
RELEVANT_THRESHOLD = 2  # standard graded->binary cutoff (TREC convention: grade >= 2 is "relevant")

UMBRELA_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        "UMBRELA_RELEVANCE",
        description=(
            "Mean UMBRELA graded relevance score (0-3) of the RAG response to the query, "
            "using UMBRELA's zero-/few-shot LLM relevance-judgment prompts (Thomas et al., "
            "'Large Language Models can Accurately Predict Searcher Preferences'). "
            "0 = not relevant at all, 3 = dedicated to the query with the exact answer."
        ),
    ),
    MeasureSpec(
        "UMBRELA_RELEVANT",
        description=(
            f"Fraction of responses graded >= {RELEVANT_THRESHOLD} by UMBRELA (the standard "
            "graded-to-binary relevance cutoff), from 0.0 to 1.0; higher is better."
        ),
    ),
    MeasureSpec(
        "UMBRELA_VALID_RESPONSE_RATE",
        description=(
            "Fraction of UMBRELA LLM judgments that were successfully parsed into a 0-3 "
            "category (vs. defaulting to 0 because no valid score could be found), from 0.0 "
            "to 1.0; a diagnostic of prompt/LLM reliability, not of response quality."
        ),
    ),
))


class UmbrelaJudge:
    """
    Grades RAG responses for query relevance using UMBRELA's LLM prompts.

    Implements LeaderboardJudgeProtocol (and optionally produces Qrels).
    Configure in workflow.yml:
        judge_class: "judges.umbrela_judge.umbrela_judge:UmbrelaJudge"
        qrels_class: "judges.umbrela_judge.umbrela_judge:UmbrelaJudge"  # optional
    """

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # UMBRELA prompt configuration
        prompt_style: str = "bing",
        few_shot: bool = False,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        """Grade each response's relevance to its topic with UMBRELA and build a leaderboard."""
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        graded = self._grade_responses(
            rag_responses=rag_responses,
            topic_titles=topic_titles,
            llm_config=llm_config,
            prompt_style=prompt_style,
            few_shot=few_shot,
        )

        builder = LeaderboardBuilder(UMBRELA_SPEC)
        for (run_id, topic_id), (grade, valid) in graded.items():
            builder.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "UMBRELA_RELEVANCE": grade,
                    "UMBRELA_RELEVANT": 1.0 if grade >= RELEVANT_THRESHOLD else 0.0,
                    "UMBRELA_VALID_RESPONSE_RATE": 1.0 if valid else 0.0,
                },
            )

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")

    def create_qrels(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        prompt_style: str = "bing",
        few_shot: bool = False,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        """Produce standard Qrels (doc_id=run_id) from the same UMBRELA grading as `judge()`."""
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        graded = self._grade_responses(
            rag_responses=rag_responses,
            topic_titles=topic_titles,
            llm_config=llm_config,
            prompt_style=prompt_style,
            few_shot=few_shot,
        )

        rows = [
            QrelRow(topic_id=topic_id, doc_id=run_id, grade=round(grade))
            for (run_id, topic_id), (grade, _valid) in graded.items()
        ]
        qrels = Qrels(rows=rows)
        qrels.verify(expected_topic_ids=expected_topic_ids, warn=True)
        return qrels

    def _grade_responses(
        self,
        rag_responses: Iterable[Report],
        topic_titles: Dict[str, str],
        llm_config: LlmConfigProtocol,
        prompt_style: str,
        few_shot: bool,
    ) -> Dict[Tuple[str, str], Tuple[float, bool]]:
        """Run one UMBRELA LLM request per response, keyed by (run_id, topic_id)."""
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []
        for i, response in enumerate(rag_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            passage = response.get_report_text()
            prompt = render_prompt(style=prompt_style, few_shot=few_shot, query=query, passage=passage)

            requests_info.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                MinimaLlmRequest(
                    request_id=f"q{i}_umbrela",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                ),
            ))

        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        llm_results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))

        graded: Dict[Tuple[str, str], Tuple[float, bool]] = {}
        for (run_id, topic_id, _), result in zip(requests_info, llm_results):
            graded[(run_id, topic_id)] = self._parse_umbrela_score(result)
        return graded

    # UMBRELA's prompts ask for the score in varied shapes depending on the
    # style/reasoning the LLM chooses, e.g. "##final score: 2", "Score: 2",
    # "final score is 2", "category 2", or just a standalone trailing digit.
    # These patterns are a close adaptation of the official UMBRELA parser
    # (umbrela.utils.common_utils.parse_fewshot_response), which scans the
    # *last* match of each pattern across the whole response so that a score
    # mentioned once in reasoning and then restated in the answer line is
    # resolved to the final (restated) value.
    _SCORE_PATTERNS: Tuple[re.Pattern, ...] = tuple(
        re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        for p in (
            r"##\s*final score\s*[:\-=]?\s*([0-3])",
            r"\"?final[ _]score\"?\s*[:\-=]?\s*([0-3])",
            r"final score is\s*([0-3])",
            r"\"?(?:o|overall|overall_score|overall score|score|o_score)\"?\s*[:\-=]\s*([0-3])",
            r"score is\s*([0-3])",
            r"score of\s+([0-3])",
            r"score:\s*([0-3])",
            r"relevance category\s*[:\-=]?\s*([0-3])",
            r"category\s*([0-3])",
            r"(?<![\d.\-])([0-3])\s*$",
        )
    )

    def _parse_umbrela_score(self, result: Any) -> Tuple[float, bool]:
        """Parse an UMBRELA LLM response into a (0-3 grade, was_valid) pair.

        Scans the full response text (not just the last line, since UMBRELA's
        "bing"-style prompt produces multi-paragraph step-by-step reasoning
        before the final score) for the *last* occurrence of any recognized
        score pattern, matching UMBRELA's own parsing strategy. If no pattern
        matches -- an empty/garbled response, an LLM error, or a value
        outside 0-3 -- the response is scored as 0 with `valid=False`,
        consistent with UMBRELA's "invalid response" handling (never crash,
        never guess from unrelated numbers).
        """
        try:
            if not isinstance(result, MinimaLlmResponse):
                print(f"[UmbrelaJudge] LLM error: {result}")
                return 0.0, False

            text = result.text
            if not text or not text.strip():
                return 0.0, False

            for pattern in self._SCORE_PATTERNS:
                last_match = None
                for m in pattern.finditer(text):
                    last_match = m
                if last_match is not None:
                    return float(last_match.group(1)), True

            print(f"[UmbrelaJudge] Could not parse UMBRELA score from response: {text!r}")
            return 0.0, False
        except Exception as exc:
            print(f"[UmbrelaJudge] Error parsing LLM response: {exc}")
            return 0.0, False
