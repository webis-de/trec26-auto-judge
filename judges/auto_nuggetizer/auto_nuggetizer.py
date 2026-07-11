#!/usr/bin/env python3
import os
import sys
from typing import Any, Dict, Optional, Sequence, Type
from pathlib import Path

cache_dir = os.environ.get("CACHE_DIR")
if cache_dir is None:
    sys.exit("ERROR: CACHE_DIR is not set.")

Path(cache_dir).mkdir(parents=True, exist_ok=True)

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


class AutoNuggetizer(AutoJudge):
    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def group_by_topic(self, rag_responses: Sequence[Report]):
        ret = {}
        for r in rag_responses:
            if r.metadata.narrative_id not in ret:
                ret[r.metadata.narrative_id] = []
            ret[r.metadata.narrative_id].append(r)
        return ret

    def extract_nuggets_for_topic(self, topic, responses, llm_config):
        from nuggetizer.models.nuggetizer import Nuggetizer
        from nuggetizer.core.types import Document, Query, Request as NuggetizerRequest

        print("extract_nuggets_for_topic called")
        doc_id_to_doc = {}

        for response in responses:
            for sentence in response.responses:
                for citation in sentence.citations:
                    doc_id = response.references[citation]
                    doc_id_to_doc[doc_id] = response.documents[doc_id]

        #nuggets = []

        documents = [
            Document(docid=doc_id, segment=doc.get_document_text())
            for doc_id, doc in doc_id_to_doc.items()
        ]

        #documents = documents[:40]

        request = NuggetizerRequest(
            query=Query(
                qid=topic.request_id,
                text=topic.title,
            ),
            documents=documents,
        )

        nuggetizer = Nuggetizer(
            model="qwen3-coder-next",
            api_keys=[llm_config.api_key],
            api_base=llm_config.base_url,
            api_type="openrouter",
)

        scored_nuggets = nuggetizer.create(request)

        if not scored_nuggets:
            raise RuntimeError(f"No nuggets created for topic {topic.request_id}")

        nuggets = [
            NuggetQuestion(
                query_id=topic.request_id,
                question=nugget.text,
            )
            for nugget in scored_nuggets
        ]

        ret = NuggetBank(query_id=topic.request_id, title_query=topic.title)
        ret.add_nuggets(nuggets)

        print(f"Created nugget bank for {topic.request_id} with {len(nuggets)} nuggets")

        return ret
    
    def extract_nugget_for_topics_fail_safe(self, topic, llm_config, topic_to_responses):
        max_attempts = 6

        for attempt in range(max_attempts):
            try:
                return self.extract_nuggets_for_topic(
                    topic,
                    topic_to_responses[topic.request_id],
                    llm_config,
                )

            except Exception as e:
                msg = str(e).lower()

                is_rate_limit = (
                    "rate limit" in msg
                    or "too many requests" in msg
                    or "429" in msg
                )

                if is_rate_limit and attempt < max_attempts - 1:
                    wait_seconds = 30 * 60
                    print(
                        f"RATE LIMIT for topic {topic.request_id}."
                        f"Waiting {wait_seconds} seconds before retry..."
                    )
                    time.sleep(wait_seconds)
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

        for topic in rag_topics:
            ret.append(self.extract_nugget_for_topics_fail_safe(topic, llm_config, topic_to_responses))  

        print(f"Returning {len(ret)} nugget banks")
        return NuggetBanks.from_banks_list(ret)


if __name__ == '__main__':
    auto_judge_to_click_command(ArgumentNuggetizer(), "argument-nuggetizer")()
