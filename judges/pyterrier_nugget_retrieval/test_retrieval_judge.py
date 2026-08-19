from types import SimpleNamespace

import pytest

from judges.pyterrier_nugget_retrieval import retrieval_judge


def nugget_banks(topic_id="topic", questions=("first nugget",)):
    nuggets = {
        question: SimpleNamespace(question=question) for question in questions
    }
    return SimpleNamespace(
        banks={
            topic_id: SimpleNamespace(nugget_bank=nuggets),
        }
    )


def test_get_nugget_queries_returns_every_nugget():
    banks = nugget_banks(questions=(" first nugget ", "second nugget"))

    assert retrieval_judge.get_nugget_queries(banks, "topic") == [
        "first nugget",
        "second nugget",
    ]


@pytest.mark.parametrize(
    ("banks", "message"),
    [
        (None, "requires nugget banks"),
        (SimpleNamespace(banks={}), "has no nugget bank"),
        (nugget_banks(questions=()), "has no nuggets"),
        (nugget_banks(questions=(" ",)), "contains an empty nugget"),
    ],
)
def test_get_nugget_queries_rejects_missing_nuggets(banks, message):
    with pytest.raises(ValueError, match=message):
        retrieval_judge.get_nugget_queries(banks, "topic")


def test_aggregate_scores_calculates_mean_min_and_max():
    assert retrieval_judge.aggregate_scores([0.25, 0.5, 1.0]) == {
        "mean": pytest.approx(7 / 12),
        "min": 0.25,
        "max": 1.0,
    }


def test_aggregate_scores_rejects_empty_input():
    with pytest.raises(ValueError, match="empty list"):
        retrieval_judge.aggregate_scores([])


def test_judge_queries_every_nugget_and_aggregates_scores(monkeypatch):
    search_calls = []

    class FakeResults:
        def __init__(self, rows):
            self.rows = rows

        def iterrows(self):
            return enumerate(self.rows)

    class FakeRetriever:
        def __init__(self, index, wmodel):
            self.wmodel = wmodel

        def search(self, query):
            search_calls.append((self.wmodel, query))
            scores = {
                "first nugget": {
                    "system-a": 10.0,
                    "system-b": 5.0,
                    "system-c": 0.0,
                },
                "second nugget": {
                    "system-a": 4.0,
                    "system-b": 8.0,
                    "system-c": 6.0,
                },
            }[query]
            return FakeResults(
                [
                    {"docno": system, "score": score}
                    for system, score in scores.items()
                ]
            )

    class FakeIndexer:
        def __init__(self, *args, **kwargs):
            pass

        def index(self, docs):
            return docs

    class FakeTokeniser:
        def getTokens(self, text):
            return text.split()

    class FakeTokeniserFactory:
        @staticmethod
        def getTokeniser():
            return FakeTokeniser()

    class FakeBuilder:
        def __init__(self):
            self.entries = []

        def add(self, **entry):
            self.entries.append(entry)

        def build(self):
            return self

    class FakeVerification:
        def __init__(self, *args, **kwargs):
            pass

        def all(self):
            return None

    fake_builder = FakeBuilder()
    fake_pt = SimpleNamespace(
        java=SimpleNamespace(autoclass=lambda _: FakeTokeniserFactory),
        IterDictIndexer=FakeIndexer,
        IndexingType=SimpleNamespace(MEMORY="memory"),
        terrier=SimpleNamespace(Retriever=FakeRetriever),
    )
    monkeypatch.setattr(retrieval_judge, "pt", fake_pt)
    monkeypatch.setattr(
        retrieval_judge, "ensure_pyterrier_is_loaded", lambda: None
    )
    monkeypatch.setattr(
        retrieval_judge, "LeaderboardBuilder", lambda _: fake_builder
    )
    monkeypatch.setattr(
        retrieval_judge, "LeaderboardVerification", FakeVerification
    )

    reports = [
        SimpleNamespace(
            metadata=SimpleNamespace(run_id=system, topic_id="topic"),
            get_report_text=lambda system=system: f"Response from {system}",
        )
        for system in ("system-a", "system-b", "system-c")
    ]

    result = retrieval_judge.RetrievalJudge().judge(
        rag_responses=reports,
        rag_topics=[],
        llm_config=None,
        nugget_banks=nugget_banks(
            questions=("first nugget", "second nugget")
        ),
    )

    assert result is fake_builder
    assert len(search_calls) == len(retrieval_judge.WEIGHTING_MODELS) * 2
    for wmodel in retrieval_judge.WEIGHTING_MODELS:
        assert (wmodel, "first nugget") in search_calls
        assert (wmodel, "second nugget") in search_calls

    entries = {entry["run_id"]: entry for entry in fake_builder.entries}
    expected = {
        "system-a": {"mean": 0.5, "min": 0.0, "max": 1.0},
        "system-b": {"mean": 0.75, "min": 0.5, "max": 1.0},
        "system-c": {"mean": 0.25, "min": 0.0, "max": 0.5},
    }
    for system, aggregations in expected.items():
        assert entries[system]["topic_id"] == "topic"
        for wmodel in retrieval_judge.WEIGHTING_MODELS:
            for aggregation, value in aggregations.items():
                assert entries[system]["values"][
                    f"{wmodel}_{aggregation}"
                ] == pytest.approx(value)
