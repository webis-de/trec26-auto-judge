from typing import List

import pytest

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData, RagtimeReportSentence
from minima_llm import MinimaLlmResponse

from judges.staged_relevance_judge.staged_relevance_judge import (
    StagedRelevanceJudge,
    STAGED_RELEVANCE_SPEC,
    _STAGE_EXPERT,
    _STAGE_FIRST_SENTENCE,
    _STAGE_OVERALL,
)


def make_report(run_id: str, topic_id: str, text: str) -> Report:
    return Report(
        metadata=ReportMetaData(team_id="team", run_id=run_id, topic_id=topic_id),
        responses=[RagtimeReportSentence(text=text)],
    )


def make_topic(topic_id: str, title: str) -> Request:
    return Request(request_id=topic_id, title=title)


def test_spec_declares_all_stage_and_cascade_measures():
    measure_names = [m.name for m in STAGED_RELEVANCE_SPEC.measures]
    for expected in (
        "FIRST_SENTENCE_RELEVANT",
        "OVERALL_RELEVANT",
        "EXPERT_ENDORSED",
        "CASCADE_IMPATIENT_USER",
        "CASCADE_CASUAL_READER_USER",
        "CASCADE_EXPERT_USER",
        "CASCADE_BALANCED_USER",
    ):
        assert expected in measure_names


def test_normalized_weights_sum_to_one():
    weights = StagedRelevanceJudge._normalized_weights(0.5, 0.3, 0.2)
    assert sum(weights) == 1.0
    assert weights == (0.5, 0.3, 0.2)

    # Un-normalized inputs get scaled proportionally.
    weights = StagedRelevanceJudge._normalized_weights(1, 1, 2)
    assert sum(weights) == 1.0
    assert weights == (0.25, 0.25, 0.5)


def test_cascade_score_requires_all_prior_stages_to_be_satisfied():
    weights = (0.5, 0.3, 0.2)

    # All stages fully graded (3/3) -> full score (3.0).
    assert StagedRelevanceJudge._cascade_score(3.0, 3.0, 3.0, weights) == 3.0

    # First sentence not relevant at all -> no credit anywhere, even if later
    # stages are fully graded.
    assert StagedRelevanceJudge._cascade_score(0.0, 3.0, 3.0, weights) == 0.0

    # Fully relevant first sentence and overall, but not expert-endorsed ->
    # only the first two stop-weights are credited (rescaled to 0-3).
    score = StagedRelevanceJudge._cascade_score(3.0, 3.0, 0.0, weights)
    assert score == 3.0 * (weights[0] + weights[1])

    # Partial (2/3) grades at every stage.
    score = StagedRelevanceJudge._cascade_score(2.0, 2.0, 2.0, weights)
    n = 2.0 / 3.0
    expected = 3.0 * (weights[0] * n + weights[1] * n * n + weights[2] * n * n * n)
    assert score == expected


class NotAResponse:
    """A stand-in for a non-MinimaLlmResponse error/failure object."""


@pytest.mark.parametrize(
    "text,expected",
    [
        # Bare score, no reasoning at all.
        ("0", 0.0),
        ("1", 1.0),
        ("2", 2.0),
        ("3", 3.0),
        # Bare decimal score.
        ("3.0", 3.0),
        ("2.0", 2.0),
        # Reasoning on earlier line(s), score alone on the final line.
        ("The text does not mention the topic at all.\n0", 0.0),
        ("This is somewhat related to the query.\n1", 1.0),
        ("The response addresses most of the query.\n2", 2.0),
        ("This is highly relevant and directly answers the query.\n3", 3.0),
        # Multi-line reasoning before the final score-only line.
        ("First I consider the query.\nThen I consider the text.\nOverall it is relevant.\n3", 3.0),
        # Leading/trailing blank lines and surrounding whitespace are tolerated.
        ("\n\nSome reasoning.\n\n2\n\n", 2.0),
        ("Reasoning text.\n   3   ", 3.0),
        # A "Score:" (or similar) label before the number is tolerated.
        ("Score: 3", 3.0),
        ("Score: 2.0", 2.0),
        ("Score:3", 3.0),
        ("score - 1", 1.0),
        ("SCORE: 0", 0.0),
        ("Reasoning above.\nScore: 2", 2.0),
        # Trailing explanation after the score on the same line is tolerated.
        ("2 - mostly relevant", 2.0),
        ("2- mostly relevant", 2.0),
        ("2.0 - mostly relevant", 2.0),
        ("3 out of 3", 3.0),
        ("1, because it is only marginally relevant", 1.0),
    ],
)
def test_parse_relevance_accepts_leading_score_with_optional_label_or_trailing_text(text, expected):
    judge = StagedRelevanceJudge()
    assert judge._parse_relevance(MinimaLlmResponse(request_id="x", text=text)) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        # Real-world examples: the LLM ignores the "final line" instruction
        # and appends the score to the end of its reasoning, on one line.
        (
            "The answer lists several reasons that people might be boycotting Starbucks, directly "
            "addressing the question. It provides a concise explanation of potential motivations, "
            "making it highly relevant. 3",
            3.0,
        ),
        (
            "The text directly explains a key way African rulers contributed to the triangular "
            "trade\u2014by capturing and selling people to European traders, which is central to the "
            "question. 3",
            3.0,
        ),
        (
            "The answer is somewhat simplified and lacks the nuance and depth a seasoned expert would "
            "provide, so it differs from what I would write. 1",
            1.0,
        ),
        # Standalone trailing score after a sentence-ending period, without a "Score" label.
        ("This is clearly on topic. 2", 2.0),
        ("Not relevant at all. 0", 0.0),
    ],
)
def test_parse_relevance_accepts_trailing_score_after_reasoning_on_same_line(text, expected):
    judge = StagedRelevanceJudge()
    assert judge._parse_relevance(MinimaLlmResponse(request_id="x", text=text)) == expected


@pytest.mark.parametrize(
    "text",
    [
        # Out-of-range "scores".
        "4",
        "-1",
        "Score: 4",
        # Non-numeric "scores".
        "three",
        # Score not at the start, nor standalone at the end, of the line
        # (buried in the middle of reasoning text, or glued to other text).
        "The score is 1.",
        "I read page 3 of the document and found nothing relevant.",
        "The query has 2 keywords but the response ignores them.",
        # A negative number must not be misparsed via its trailing digit(s).
        "The final adjustment was -1",
        # Garbled / unparseable responses.
        "unparseable response",
        "I cannot judge this.",
        # Empty / whitespace-only responses.
        "",
        "   ",
        "\n\n\n",
    ],
)
def test_parse_relevance_returns_zero_on_malformed_output(text):
    judge = StagedRelevanceJudge()
    assert judge._parse_relevance(MinimaLlmResponse(request_id="x", text=text)) == 0.0



def test_parse_relevance_returns_zero_for_non_response_error():
    judge = StagedRelevanceJudge()
    assert judge._parse_relevance(NotAResponse()) == 0.0
    assert judge._parse_relevance("some non-response error string") == 0.0
    assert judge._parse_relevance(None) == 0.0
    assert judge._parse_relevance(Exception("timeout")) == 0.0


def test_parse_relevance_returns_zero_when_text_attribute_is_not_a_string():
    """If `result.text` is an unexpected type (e.g. None), parsing must fail
    closed (return 0) instead of raising."""
    judge = StagedRelevanceJudge()
    response = MinimaLlmResponse(request_id="x", text=None)
    assert judge._parse_relevance(response) == 0.0


def test_judge_combines_three_llm_calls_per_response(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    reports: List[Report] = [
        make_report("run1", "t1", "relevant text"),
    ]
    topics = [make_topic("t1", "some query")]

    # Map each stage request_id suffix to a canned graded LLM answer (reasoning
    # followed by a final line containing only the 0-3 score).
    answers = {
        _STAGE_FIRST_SENTENCE: "The first sentence directly addresses the query.\n3",
        _STAGE_OVERALL: "The response as a whole is on topic.\n3",
        _STAGE_EXPERT: "An expert would have written this quite differently.\n0",
    }

    async def fake_run_batched(self, requests):
        results = []
        for req in requests:
            stage = next(s for s in answers if req.request_id.endswith(f"_{s}"))
            results.append(MinimaLlmResponse(request_id=req.request_id, text=answers[stage]))
        return results

    monkeypatch.setattr(
        "judges.staged_relevance_judge.staged_relevance_judge.OpenAIMinimaLlm.run_batched",
        fake_run_batched,
    )

    class FakeLlmConfig:
        raw = None

    leaderboard = StagedRelevanceJudge().judge(
        rag_responses=reports,
        rag_topics=topics,
        llm_config=FakeLlmConfig(),
    )

    entry = leaderboard.entries[0]
    assert entry.values["FIRST_SENTENCE_RELEVANT"] == 3.0
    assert entry.values["OVERALL_RELEVANT"] == 3.0
    assert entry.values["EXPERT_ENDORSED"] == 0.0
    # s1=3/3=1, s2=3/3=1, s3=0/3=0 -> n1=1, n1*n2=1, n1*n2*n3=0
    # CASCADE_IMPATIENT_USER:      3 * (0.7*1 + 0.2*1 + 0.1*0) == 2.7
    # CASCADE_CASUAL_READER_USER:  3 * (0.2*1 + 0.6*1 + 0.2*0) == 2.4
    # CASCADE_EXPERT_USER:         3 * (0.1*1 + 0.2*1 + 0.7*0) == 0.9
    # CASCADE_BALANCED_USER:       3 * (1/3*1 + 1/3*1 + 1/3*0) == 2.0
    assert entry.values["CASCADE_IMPATIENT_USER"] == pytest.approx(2.7)
    assert entry.values["CASCADE_CASUAL_READER_USER"] == pytest.approx(2.4)
    assert entry.values["CASCADE_EXPERT_USER"] == pytest.approx(0.9)
    assert entry.values["CASCADE_BALANCED_USER"] == pytest.approx(2.0)
