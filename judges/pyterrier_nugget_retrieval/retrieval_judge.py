#!/usr/bin/env python3
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type

from autojudge_base import (
    Report,
    LeaderboardSpec,
    LeaderboardBuilder,
    LeaderboardVerification,
    MeasureSpec,
    AutoJudge,
    auto_judge_to_click_command,
    Leaderboard,
    Qrels,
    Request,
    LlmConfigProtocol,
    NuggetBanks,
    NuggetBanksProtocol,
)
from collections import defaultdict
from tqdm import tqdm
from tira.third_party_integrations import ensure_pyterrier_is_loaded
import pyterrier as pt


def group_by_topic_id(rag_responses: Sequence[Report]) -> Dict[str, Dict[str, str]]:
    """Group RAG responses by topic_id, then by run_id."""
    ret: Dict[str, Dict[str, str]] = defaultdict(dict)
    for rag_response in rag_responses:
        run_id: str = rag_response.metadata.run_id
        topic_id: str = rag_response.metadata.topic_id
        ret[topic_id][run_id] = rag_response.get_report_text()
    return ret


WEIGHTING_MODELS = (
    "BM25",
    "DirichletLM",
    "Hiemstra_LM",
    "DFIC",
    "DPH",
    "DLH",
    "Tf",
    "TF_IDF",
    "PL2",
    "InL2",
)

LEADERBOARD_SPEC = LeaderboardSpec(
    measures=tuple(
        MeasureSpec(
            f"{wmodel}_{aggregation}",
            description=description.format(wmodel=wmodel),
        )
        for wmodel in WEIGHTING_MODELS
        for aggregation, description in (
            (
                "mean",
                "Mean per-nugget normalized {wmodel} retrieval score, from 0.0 "
                "to 1.0; higher means the response covers the nugget bank more "
                "consistently.",
            ),
            (
                "min",
                "Minimum per-nugget normalized {wmodel} retrieval score, from "
                "0.0 to 1.0; higher means even the least-covered nugget is well "
                "represented.",
            ),
            (
                "max",
                "Maximum per-nugget normalized {wmodel} retrieval score, from "
                "0.0 to 1.0; higher means at least one nugget is strongly "
                "represented.",
            ),
        )
    )
)


def get_nugget_queries(
    nugget_banks: Optional[NuggetBanksProtocol], topic_id: str
) -> List[str]:
    if nugget_banks is None:
        raise ValueError("This judge requires nugget banks as input.")

    try:
        nugget_bank = nugget_banks.banks[topic_id]
    except KeyError as error:
        raise ValueError(f"Topic {topic_id!r} has no nugget bank.") from error

    nuggets = list(nugget_bank.nugget_bank.values())
    if not nuggets:
        raise ValueError(f"Topic {topic_id!r} has no nuggets.")

    queries: List[str] = []
    for nugget in nuggets:
        question = getattr(nugget, "question", None)
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Topic {topic_id!r} contains an empty nugget.")
        queries.append(question.strip())
    return queries


def aggregate_scores(scores: Sequence[float]) -> Dict[str, float]:
    if not scores:
        raise ValueError("Cannot aggregate an empty list of nugget scores.")
    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
    }


class RetrievalJudge(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        return None

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Optional[Qrels]:
        return None

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        # Standard output path settings (auto-filled by judge_runner)
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        ensure_pyterrier_is_loaded()
        tokeniser: Any = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

        def pt_tokenize(text: str) -> str:
            return " ".join(tokeniser.getTokens(text))

        topic_id_to_responses: Dict[str, Dict[str, str]] = group_by_topic_id(
            rag_responses
        )
        all_systems: set[str] = set(i.metadata.run_id for i in rag_responses)
        topic_id_to_nuggets: Dict[str, List[str]] = {
            topic_id: get_nugget_queries(nugget_banks, topic_id)
            for topic_id in topic_id_to_responses
        }

        builder: LeaderboardBuilder = LeaderboardBuilder(LEADERBOARD_SPEC)

        for topic in tqdm(topic_id_to_responses.keys(), "Process Topics"):
            docs: List[Dict[str, str]] = [
                {"docno": system, "text": system_response}
                for system, system_response in topic_id_to_responses[topic].items()
            ]
            system_to_measure_to_score: Dict[str, Dict[str, float]] = defaultdict(
                dict
            )
            index: Any = pt.IterDictIndexer(
                "/not-needed/for-memory-index",
                meta={"docno": 100},
                type=pt.IndexingType.MEMORY,
            ).index(docs)

            for wmodel in WEIGHTING_MODELS:
                retriever: Any = pt.terrier.Retriever(index, wmodel=wmodel)
                system_to_nugget_scores: Dict[str, List[float]] = {
                    system: [] for system in all_systems
                }

                for nugget in topic_id_to_nuggets[topic]:
                    results: Any = retriever.search(pt_tokenize(nugget))
                    raw_scores: Dict[str, float] = {
                        row["docno"]: float(row["score"])
                        for _, row in results.iterrows()
                    }
                    lowest_score = min(raw_scores.values(), default=0.0)
                    highest_score = max(raw_scores.values(), default=0.0)

                    for system in all_systems:
                        if system not in raw_scores:
                            normalized_score = 0.0
                        elif highest_score > lowest_score:
                            normalized_score = (
                                raw_scores[system] - lowest_score
                            ) / (highest_score - lowest_score)
                        else:
                            normalized_score = 1.0
                        system_to_nugget_scores[system].append(normalized_score)

                for system, nugget_scores in system_to_nugget_scores.items():
                    for aggregation, score in aggregate_scores(
                        nugget_scores
                    ).items():
                        system_to_measure_to_score[system][
                            f"{wmodel}_{aggregation}"
                        ] = score

            for system in all_systems:
                builder.add(
                    run_id=system,
                    topic_id=topic,
                    values=system_to_measure_to_score[system],
                )

        leaderboard: Leaderboard = builder.build()
        LeaderboardVerification(leaderboard, on_missing="fix_aggregate").all()
        return leaderboard


if __name__ == '__main__':
    auto_judge_to_click_command(RetrievalJudge(), "retrieval-judge")()
