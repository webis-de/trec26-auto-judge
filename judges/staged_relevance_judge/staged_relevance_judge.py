#!/usr/bin/env python3
"""
StagedRelevanceJudge: LLM-based judge with preconfigured cascade "user models".

Extends the TinyJudge idea (ask an LLM whether the first sentence of a
response is relevant) with two further LLM checks:

  Stage 1 - FIRST_SENTENCE_RELEVANT: Is the first sentence of the response
            relevant to the query? (what a user sees in a snippet/preview)
  Stage 2 - OVERALL_RELEVANT: Is the response as a whole relevant to the
            query? (what a user sees after reading the full response)
  Stage 3 - EXPERT_ENDORSED: Would a long-time domain expert for this topic
            have written the response roughly the way it is, i.e. does it
            hold up to expert scrutiny? (what a very engaged/expert user
            would conclude after reading closely)

All three are graded on a 0-3 scale (0 = not at all, 3 = fully), matching
common graded relevance judgment conventions.

These three independent stage judgments are then combined into four
CASCADE_*_USER measures, one per preconfigured "user model" (see
`_CASCADE_USER_MODELS` below). Each user model assumes a population of users
split by how far they read: a fraction who give up after stage 1 (just
skimmed the first sentence), a fraction who give up after stage 2 (read the
whole response), and a fraction who go all the way to stage 3 (read it with
expert-level scrutiny). A user is only considered satisfied at stage k if
all stages up to and including k were graded highly, so the satisfaction
probability for users stopping at stage k is modeled as the product of the
first k stage grades (normalized to 0.0-1.0). Each CASCADE_*_USER score is
the stop-fraction-weighted sum of these per-stage satisfaction
probabilities, rescaled back to the 0-3 grade scale:

    n1, n2, n3 = s1 / 3, s2 / 3, s3 / 3
    CASCADE_*_USER = 3 * (w1 * n1 + w2 * n1 * n2 + w3 * n1 * n2 * n3)

where `s1`, `s2`, `s3` are the (0-3) stage grades and `w1`, `w2`, `w3` are
the fractions of users assumed to stop at each stage for that user model
(normalized to sum to 1).
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
    Report,
    Request,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm


GRADE_MAX = 3  # all judgments use a graded 0-3 scale (0=not at all, 3=fully)

# Preconfigured cascade user models: fixed (stage1, stage2, stage3) stop
# fractions representing different reader personas. Each becomes its own
# leaderboard measure, so a single run can be compared across several
# assumptions about user reading behavior without needing separate
# workflow.yml variants/re-runs.
_CASCADE_USER_MODELS: Dict[str, Tuple[float, float, float]] = {
    # Mostly bounces after skimming just the first sentence.
    "CASCADE_IMPATIENT_USER": (0.7, 0.2, 0.1),
    # Usually reads the whole response but rarely scrutinizes it like an expert.
    "CASCADE_CASUAL_READER_USER": (0.2, 0.6, 0.2),
    # Mostly reads with expert-level scrutiny (e.g. professional/research use cases).
    "CASCADE_EXPERT_USER": (0.1, 0.2, 0.7),
    # Equal weight across all three reading depths.
    "CASCADE_BALANCED_USER": (1 / 3, 1 / 3, 1 / 3),
}

STAGED_RELEVANCE_SPEC = LeaderboardSpec(measures=tuple(
    [
        MeasureSpec("FIRST_SENTENCE_RELEVANT", description="Stage 1: LLM graded judgment of first sentence relevance (0-3, 0=not relevant, 3=highly relevant)"),
        MeasureSpec("OVERALL_RELEVANT", description="Stage 2: LLM graded judgment of whole-response relevance (0-3, 0=not relevant, 3=highly relevant)"),
        MeasureSpec("EXPERT_ENDORSED", description="Stage 3: LLM graded judgment of whether a long-time domain expert would have written a similar response (0-3, 0=not at all, 3=exactly as the expert would have written it)"),
    ]
    + [
        MeasureSpec(
            name,
            description=(
                f"Cascade user-model aggregate of stages 1-3 for the '{name}' persona "
                f"(stop fractions stage1={w1:.3g}, stage2={w2:.3g}, stage3={w3:.3g}), "
                "rescaled to the same 0-3 grade scale"
            ),
        )
        for name, (w1, w2, w3) in _CASCADE_USER_MODELS.items()
    ]
))

# Stage names used consistently for request-id tagging and parsing.
_STAGE_FIRST_SENTENCE = "first_sentence"
_STAGE_OVERALL = "overall"
_STAGE_EXPERT = "expert"
_STAGES: Tuple[str, ...] = (_STAGE_FIRST_SENTENCE, _STAGE_OVERALL, _STAGE_EXPERT)


class StagedRelevanceJudge:
    """
    Judges responses with a 3-stage LLM cascade and aggregates them via four
    preconfigured cascade user models.

    Implements LeaderboardJudgeProtocol. Configure in workflow.yml:
        judge_class: "judges.staged_relevance_judge.staged_relevance_judge:StagedRelevanceJudge"
    """

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # how many leading response segments count as the "first sentence" stage
        segments: int = 1,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        """Judge responses with three LLM checks and four cascade user models."""
        weights_by_model = {
            name: self._normalized_weights(*fractions)
            for name, fractions in _CASCADE_USER_MODELS.items()
        }

        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        # Collect all requests with metadata: (run_id, topic_id, stage, request)
        requests_info: List[Tuple[str, str, str, MinimaLlmRequest]] = []
        for i, response in enumerate(rag_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            first_sentence_text = " ".join(r.text for r in response.responses[:segments] if r.text)
            full_text = response.get_report_text()

            for stage, request in self._build_stage_requests(
                request_id_prefix=f"q{i}",
                query=query,
                first_sentence_text=first_sentence_text,
                full_text=full_text,
            ):
                requests_info.append((response.metadata.run_id, response.metadata.topic_id, stage, request))

        # Run all LLM requests (all stages, all responses) in one batch.
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        llm_results = asyncio.run(backend.run_batched([req for _, _, _, req in requests_info]))

        # Regroup per-response stage scores: (run_id, topic_id) -> {stage: score}
        stage_scores: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (run_id, topic_id, stage, _), result in zip(requests_info, llm_results):
            key = (run_id, topic_id)
            stage_scores.setdefault(key, {})[stage] = self._parse_relevance(result)

        builder = LeaderboardBuilder(STAGED_RELEVANCE_SPEC)
        for (run_id, topic_id), scores in stage_scores.items():
            s1 = scores.get(_STAGE_FIRST_SENTENCE, 0.0)
            s2 = scores.get(_STAGE_OVERALL, 0.0)
            s3 = scores.get(_STAGE_EXPERT, 0.0)

            values = {
                "FIRST_SENTENCE_RELEVANT": s1,
                "OVERALL_RELEVANT": s2,
                "EXPERT_ENDORSED": s3,
            }
            for name, weights in weights_by_model.items():
                values[name] = self._cascade_score(s1, s2, s3, weights)

            builder.add(run_id=run_id, topic_id=topic_id, values=values)

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")

    @staticmethod
    def _normalized_weights(w1: float, w2: float, w3: float) -> Tuple[float, float, float]:
        """Normalize the three stop-fractions to sum to 1."""
        total = w1 + w2 + w3
        if total <= 0:
            raise ValueError("At least one of the three stop fractions must be > 0")
        return (w1 / total, w2 / total, w3 / total)

    @staticmethod
    def _cascade_score(s1: float, s2: float, s3: float, weights: Tuple[float, float, float]) -> float:
        """Combine 0-3 graded stage scores via the cascade user model.

        A user who stops at stage k is only "satisfied" if all stages up to
        and including k were relevant/endorsed, so the stop probability at
        stage k is weighted by the product of the first k stage scores.
        Grades are normalized to 0.0-1.0 for the cascade multiplication, then
        the result is rescaled back to the 0-3 grade scale.
        """
        w1, w2, w3 = weights
        n1, n2, n3 = s1 / GRADE_MAX, s2 / GRADE_MAX, s3 / GRADE_MAX
        return GRADE_MAX * (w1 * n1 + w2 * (n1 * n2) + w3 * (n1 * n2 * n3))

    @staticmethod
    def _build_stage_requests(
        request_id_prefix: str,
        query: str,
        first_sentence_text: str,
        full_text: str,
    ) -> List[Tuple[str, MinimaLlmRequest]]:
        """Build the three per-response LLM requests, one for each stage."""
        return [
            (
                _STAGE_FIRST_SENTENCE,
                MinimaLlmRequest(
                    request_id=f"{request_id_prefix}_{_STAGE_FIRST_SENTENCE}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance evaluator. Give a brief reason for your judgment, then "
                                f"end your response with a final line containing only the score and "
                                "nothing else on that line."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "How relevant is this to the query, on a scale of 0 to 3 "
                                "(0 = not relevant at all, 1 = marginally relevant, 2 = relevant, "
                                "3 = highly relevant)? Briefly give your reason. Then, on the last line, "
                                "print only the score (0, 1, 2, or 3) and nothing else.\n\n"
                                f"Query: {query}\nText: {first_sentence_text}"
                            ),
                        },
                    ],
                    temperature=0.0,
                ),
            ),
            (
                _STAGE_OVERALL,
                MinimaLlmRequest(
                    request_id=f"{request_id_prefix}_{_STAGE_OVERALL}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance evaluator. Give a brief reason for your judgment, then "
                                "end your response with a final line containing only the score and "
                                "nothing else on that line."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "How relevant is this response, as a whole, to the query, on a scale of 0 to 3 "
                                "(0 = not relevant at all, 1 = marginally relevant, 2 = relevant, "
                                "3 = highly relevant)? Briefly give your reason. Then, on the last line, "
                                "print only the score (0, 1, 2, or 3) and nothing else.\n\n"
                                f"Query: {query}\nResponse: {full_text}"
                            ),
                        },
                    ],
                    temperature=0.0,
                ),
            ),
            (
                _STAGE_EXPERT,
                MinimaLlmRequest(
                    request_id=f"{request_id_prefix}_{_STAGE_EXPERT}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a long-time domain expert on the topic of the query, with deep "
                                "expertise built over years of dealing with it. Give a brief reason for your "
                                "judgment, then end your response with a final line containing only the score "
                                "and nothing else on that line."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Imagine you have dealt with this topic for a long time and have deep expertise "
                                "in it. On a scale of 0 to 3, how close is this response to what you yourself "
                                "would have written (0 = not at all as you would have written it, "
                                "1 = somewhat different, 2 = close to how you would have written it, "
                                "3 = essentially exactly as you would have written it)? Briefly give your "
                                "reason. Then, on the last line, print only the score (0, 1, 2, or 3) and "
                                "nothing else.\n\n"
                                f"Query: {query}\nResponse: {full_text}"
                            ),
                        },
                    ],
                    temperature=0.0,
                ),
            ),
        ]

    # Matches a score at the start of the last line, optionally preceded by a
    # "Score" label (e.g. "3", "Score: 3", "Score 2.0", "2- mostly relevant",
    # "2.0 - mostly relevant"). Trailing explanation text after the number is
    # ignored; text *before* the number (other than an optional label) is not.
    _SCORE_LINE_RE = re.compile(r"^(?:score\s*[:\-]?\s*)?(\d+(?:\.\d+)?)", re.IGNORECASE)

    def _parse_relevance(self, result: Any) -> float:
        """Parse LLM response to a graded relevance score (0-3).

        Prompts instruct the LLM to reason first, then end with a final line
        containing only the score. Parsing is lenient about the exact shape
        of that last line (an optional "Score:" label, an integer or decimal
        value, and optional trailing explanation text are all tolerated), but
        the numeric score must appear at the very start of the line (after
        the optional label) and be within the valid 0-3 range. Anything else
        (missing score, score buried in the middle of reasoning text,
        out-of-range value, empty/garbled response, a non-response error
        object, ...) is treated as a parsing failure and scored as 0, rather
        than guessing from reasoning text that may itself mention unrelated
        numbers.
        """
        try:
            if not isinstance(result, MinimaLlmResponse):
                print(f"[StagedRelevanceJudge] LLM error: {result}")
                return 0.0

            text = result.text
            if not text or not text.strip():
                return 0.0

            lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
            if not lines:
                return 0.0

            last_line = lines[-1]
            match = self._SCORE_LINE_RE.match(last_line)
            if match:
                value = float(match.group(1))
                if 0.0 <= value <= GRADE_MAX:
                    return value

            print(f"[StagedRelevanceJudge] Could not parse score from last line: {last_line!r}")
            return 0.0
        except Exception as exc:
            print(f"[StagedRelevanceJudge] Error parsing LLM response: {exc}")
            return 0.0
