#!/usr/bin/env python3
import os
import sys
from typing import Any, Dict, Optional, Sequence, Type
from pathlib import Path
from tqdm import tqdm
import json


from autojudge_base import (
    AutoJudge,
    Report,
    Request as AutoJudgeRequest,
    LeaderboardSpec,
    LeaderboardBuilder,
    LeaderboardVerification,
    MeasureSpec,
    auto_judge_to_click_command,
    Leaderboard,
    Qrels,
    LlmConfigProtocol,
    NuggetBanks,
    NuggetBanksProtocol,
)


from autojudge_base.nugget_data import (
    NuggetBank,
    NuggetBanks,
    NuggetQuestion
)

import time

DESC_PREFIX = (
    "The Nuggetizer evaluates RAG responses using nuggets classified as vital, "
    "okay, or not relevant. It grades each response's coverage of a nugget as "
    "support, partial support, or not supported. "
)

AUTO_NUGGETIZER_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("vital_and_okay", description=DESC_PREFIX + "Per topic, this is the number of vital or okay nuggets receiving support or partial support. The aggregate value is the mean count across topics; higher is better."),
    MeasureSpec("vital", description=DESC_PREFIX + "Per topic, this is the number of vital nuggets receiving support or partial support. The aggregate value is the mean count across topics; higher is better."),
    MeasureSpec("vital_support", description=DESC_PREFIX + "Per topic, this is the number of vital nuggets receiving support. The aggregate value is the mean count across topics; higher is better."),
    MeasureSpec("vital_partial", description=DESC_PREFIX + "Per topic, this is the number of vital nuggets receiving partial support. The aggregate value is the mean count across topics."),
    MeasureSpec("okay_support", description=DESC_PREFIX + "Per topic, this is the number of okay nuggets receiving support. The aggregate value is the mean count across topics; higher is better."),
    MeasureSpec("okay_partial", description=DESC_PREFIX + "Per topic, this is the number of okay nuggets receiving partial support. The aggregate value is the mean count across topics."),
    MeasureSpec("okay", description=DESC_PREFIX + "Per topic, this is the number of okay nuggets receiving support or partial support. The aggregate value is the mean count across topics; higher is better."),

    MeasureSpec("percentage_vital_and_okay", description=DESC_PREFIX + "Fraction of vital or okay nuggets receiving support or partial support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics; higher is better."),
    MeasureSpec("percentage_vital", description=DESC_PREFIX + "Fraction of vital nuggets receiving support or partial support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics; higher is better."),
    MeasureSpec("percentage_vital_support", description=DESC_PREFIX + "Fraction of vital nuggets receiving support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics; higher is better."),
    MeasureSpec("percentage_vital_partial", description=DESC_PREFIX + "Fraction of vital nuggets receiving partial support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics."),
    MeasureSpec("percentage_okay_partial", description=DESC_PREFIX + "Fraction of okay nuggets receiving partial support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics."),
    MeasureSpec("percentage_okay_support", description=DESC_PREFIX + "Fraction of okay nuggets receiving support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics; higher is better."),
    MeasureSpec("percentage_okay", description=DESC_PREFIX + "Fraction of okay nuggets receiving support or partial support, from 0.0 to 1.0. The aggregate value is the mean fraction across topics; higher is better."),
))

class AutoNuggetizer(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def group_by_topic(self, rag_responses: Sequence[Report]):
        ret = {}
        for r in rag_responses:
            if r.metadata.narrative_id not in ret:
                ret[r.metadata.narrative_id] = []
            ret[r.metadata.narrative_id].append(r)
        return ret

    def extract_nuggets_for_topic(self, topic, responses, llm_config, use_documents, use_responses):
        from nuggetizer.models.nuggetizer import Nuggetizer
        from nuggetizer.core.types import Document, Query, Request as NuggetizerRequest

        print("extract_nuggets_for_topic called")
        doc_id_to_doc = {}

        if use_documents:
            for response in responses:
                for sentence in response.responses:
                    for citation in sentence.citations:
                        doc_id_to_doc[citation] = response.documents[citation].get_document_text()

        if use_responses:
            for i, response in zip(range(len(responses)), responses):
                doc_id_to_doc[f"response-{i}"] = response.get_report_text()

        documents = [
            Document(docid=doc_id, segment=doc)
            for doc_id, doc in doc_id_to_doc.items()
        ]

        request = NuggetizerRequest(
            query=Query(
                qid=topic.request_id,
                text=topic.title,
            ),
            documents=documents,
        )

        print(f"Run nuggetizing on {len(documents)} docs...")

        nuggetizer = Nuggetizer(
            model=llm_config.model,
            api_keys=[llm_config.api_key],
            api_base=llm_config.base_url,
            api_type="openrouter",
        )

        scored_nuggets = nuggetizer.create(request)

        nuggets = [
            NuggetQuestion(
                query_id=topic.request_id,
                question=nugget.text,
                importance=nugget.importance
            )
            for nugget in scored_nuggets
        ]

        ret = NuggetBank(query_id=topic.request_id, title_query=topic.title)
        ret.add_nuggets(nuggets)

        print(f"Created nugget bank for {topic.request_id} with {len(nuggets)} nuggets")

        return ret
    
    def extract_nugget_for_topics_fail_safe(self, topic, llm_config, topic_to_responses, use_documents, use_responses):
        from nuggetizer.models.nuggetizer import Nuggetizer
        from nuggetizer.core.types import Document, Query, Request as NuggetizerRequest

        max_attempts = 6

        for attempt in range(max_attempts):
            try:
                return self.extract_nuggets_for_topic(
                    topic,
                    topic_to_responses[topic.request_id],
                    llm_config,
                    use_documents,
                    use_responses
                )

            except Exception as e:
                if attempt < max_attempts - 1:
                    print("retry...")
                    time.sleep(20)
                    continue

                print(f"FAILED topic {topic.request_id} ({topic.title}): {e}")
                raise


    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[AutoJudgeRequest],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        corpus: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[NuggetBanksProtocol]:
        topic_to_responses = self.group_by_topic(rag_responses)
        ret = []

        assert kwargs["use_documents"] or kwargs["use_responses"]

        for topic in tqdm(rag_topics, "Process topics"):
            ret.append(self.extract_nugget_for_topics_fail_safe(topic, llm_config, topic_to_responses, kwargs["use_documents"], kwargs["use_responses"]))

        print(f"Returning {len(ret)} nugget banks")
        return NuggetBanks.from_banks_list(ret)

    def assign_nuggets(self, topic_id, query, response, nuggets, llm_config, out):
        from nuggetizer.models.nuggetizer import Nuggetizer
        from nuggetizer.core.types import ScoredNugget
        reformatted_nuggets = []

        for nugget in nuggets:
            reformatted_nuggets.append(ScoredNugget(text=nugget.question, importance=nugget.importance))

        nuggetizer = Nuggetizer(
            model=llm_config.model,
            api_keys=[llm_config.api_key],
            api_base=llm_config.base_url,
            api_type="openrouter",
        )

        max_attempts = 6

        for attempt in range(max_attempts):
            try:
                assigned = nuggetizer.assign(query, response.get_report_text(), reformatted_nuggets)
                continue
            except:
                print("retry...")
                time.sleep(10)
        vital_count, okay_count = 0, 0
        vital_support, vital_partial = 0, 0
        okay_support, okay_partial = 0, 0

        with open(out, "a+") as f:
            for l in assigned:
                l = {"topic_id": topic_id, "response_team": response.metadata.team_id, "response_run": response.metadata.run_id, "nugget": l.text, "importance": l.importance, "assignment": l.assignment}
                f.write(json.dumps(l) + "\n")

                if l["assignment"] not in ("partial_support", "support", "not_support"):
                    raise ValueError(f"Unexpected assignment: {l['assignment']}")

                if l["importance"] == "okay":
                    okay_count += 1
                    if l["assignment"] == "support":
                        vital_support += 1
                    elif l["assignment"] == "partial_support":
                        vital_partial += 1
                elif l["importance"] == "vital":
                    vital_count += 1
                    if l["assignment"] == "support":
                        okay_support += 1
                    elif l["assignment"] == "partial":
                        okay_partial += 1
                else:
                    raise ValueError("sadsa")
                
        return {
            "vital_and_okay": vital_support + vital_partial + okay_support + okay_partial,
            "vital": vital_support + vital_partial,
            "vital_support": vital_support,
            "vital_partial": vital_partial,
            "okay_support": okay_support,
            "okay_partial": okay_partial,
            "okay": okay_support + okay_partial,

            "percentage_vital_and_okay": (vital_support + vital_partial + okay_support + okay_partial)  / (vital_count + okay_count) if (vital_count + okay_count) > 0 else 0,
            "percentage_vital": (vital_support + vital_partial) / vital_count if vital_count > 0 else 0,
            "percentage_vital_support": vital_support / vital_count if vital_count > 0 else 0,
            "percentage_vital_partial": vital_partial / vital_count if vital_count > 0 else 0,
            "percentage_okay_support": okay_support / okay_count if okay_count > 0 else 0,
            "percentage_okay_partial": okay_partial / okay_count if okay_count > 0 else 0,
            "percentage_okay": (okay_support + okay_partial) / okay_count if okay_count > 0 else 0,
        }

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        filebase: str = "default",
        outdir: Path = Path("."),
        **kwargs: Any,
    ) -> Leaderboard:
        topic_to_responses = self.group_by_topic(rag_responses)
        topic_titles: Dict[str, str] = {t.request_id: t.title or "" for t in rag_topics}

        builder = LeaderboardBuilder(AUTO_NUGGETIZER_SPEC)

        for topic in topic_to_responses.keys():
            query = topic_titles[topic]
            nuggets = list(nugget_banks.banks[topic].nugget_bank.values())

            for response in topic_to_responses[topic]:
                preds = self.assign_nuggets(topic, query, response, nuggets, llm_config, Path(filebase + "-assignments.jsonl") )
                builder.add(run_id=response.metadata.run_id, topic_id=topic, values=preds)

        return builder.build(expected_topic_ids=topic_titles.keys(), on_missing="fix_aggregate")