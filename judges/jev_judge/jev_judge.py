#!/usr/bin/env python3
"""JevJudge: TinyJudge-style first-sentence relevance judge backed by the jev API."""

import asyncio
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
from typesafe_sdk import AsyncTypeSafeClient, Noul, Score


JEV_SPEC = LeaderboardSpec(measures=(
    MeasureSpec(
        "FIRST_SENTENCE_RELEVANT",
        description="jev probability (0.0-1.0) that the first sentence of the response is relevant to the query",
    ),
    MeasureSpec(
        "FIRST_SENTENCE_RELEVANCE_GRADE",
        description="jev expected relevance grade (0.0-3.0) of the first sentence for the query (0=not relevant, 3=highly relevant)",
    ),
))

_QUESTIONS = {
    "relevant": Noul(instructions="Is the text relevant to the query?"),
    "grade": Score(
        instructions="How relevant is the text to the query?",
        criteria=[
            "not relevant at all",
            "marginally relevant",
            "relevant",
            "highly relevant",
        ],
    ),
}


class JevJudge:
    """Implements LeaderboardJudgeProtocol using the jev API (needs TYPESAFE_API_KEY)."""

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        segments: int = 1,
        max_concurrency: int = 8,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}

        items: List[Tuple[str, str, str]] = []  # (run_id, topic_id, document)
        for response in rag_responses:
            query = topic_titles.get(response.metadata.topic_id, "")
            text = " ".join(r.text for r in response.responses[:segments] if r.text)
            items.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                f"Query: {query}\nText: {text}",
            ))

        results = asyncio.run(self._run_all([doc for _, _, doc in items], max_concurrency))

        builder = LeaderboardBuilder(JEV_SPEC)
        for (run_id, topic_id, _), values in zip(items, results):
            builder.add(run_id=run_id, topic_id=topic_id, values=values)

        return builder.build(expected_topic_ids=list(topic_titles.keys()), on_missing="fix_aggregate")

    @staticmethod
    async def _run_all(documents: List[str], max_concurrency: int) -> List[Dict[str, float]]:
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async with AsyncTypeSafeClient() as client:

            async def one(document: str) -> Dict[str, float]:
                async with semaphore:
                    try:
                        response = await client.system_one(
                            state={"document": document},
                            questions=_QUESTIONS,
                        )
                        return {
                            "FIRST_SENTENCE_RELEVANT": float(response.nouls["relevant"].noul),
                            "FIRST_SENTENCE_RELEVANCE_GRADE": float(response.scores["grade"].score),
                        }
                    except Exception as e:  # a failed item counts as not relevant
                        print(f"jev request failed: {e!r}")
                        return {"FIRST_SENTENCE_RELEVANT": 0.0, "FIRST_SENTENCE_RELEVANCE_GRADE": 0.0}

            return list(await asyncio.gather(*(one(d) for d in documents)))
