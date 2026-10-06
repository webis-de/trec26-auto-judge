from typing import List

import pytest

from autojudge_base import Request
from autojudge_base.report import Report, ReportMetaData, RagtimeReportSentence
from minima_llm import MinimaLlmResponse

from judges.umbrela_judge.umbrela_judge import UmbrelaJudge, UMBRELA_SPEC
from judges.umbrela_judge.prompts import get_prompt_template, render_prompt


def make_report(run_id: str, topic_id: str, text: str) -> Report:
    return Report(
        metadata=ReportMetaData(team_id="team", run_id=run_id, topic_id=topic_id),
        responses=[RagtimeReportSentence(text=text)],
    )


def make_topic(topic_id: str, title: str) -> Request:
    return Request(request_id=topic_id, title=title)


def test_spec_declares_all_measures():
    measure_names = [m.name for m in UMBRELA_SPEC.measures]
    for expected in ("UMBRELA_RELEVANCE", "UMBRELA_RELEVANT", "UMBRELA_VALID_RESPONSE_RATE"):
        assert expected in measure_names


def test_prompt_templates_contain_placeholders():
    for style, few_shot in (("bing", False), ("bing", True), ("basic", False)):
        template = get_prompt_template(style=style, few_shot=few_shot)
        assert "{query}" in template
        assert "{passage}" in template


def test_render_prompt_fills_placeholders():
    prompt = render_prompt("bing", False, query="my query", passage="my passage")
    assert "my query" in prompt
    assert "my passage" in prompt
    assert "{query}" not in prompt
    assert "{passage}" not in prompt


def test_unsupported_style_raises():
    with pytest.raises(ValueError):
        get_prompt_template(style="unknown")


def test_fewshot_basic_not_supported():
    with pytest.raises(ValueError):
        get_prompt_template(style="basic", few_shot=True)


class NotAResponse:
    """A stand-in for a non-MinimaLlmResponse error/failure object."""


@pytest.mark.parametrize(
    "text,expected_grade,expected_valid",
    [
        ("Some brief reasoning about relevance.\n##final score: 3", 3.0, True),
        ("Reasoning...\n##final score: 0", 0.0, True),
        ("Score: 2", 2.0, True),
        ("The final score is 1", 1.0, True),
        ("Reasoning text.\nrelevance category: 2", 2.0, True),
        ("Reasoning only, score at end 3", 3.0, True),
        ("category 1 fits best", 1.0, True),
    ],
)
def test_parse_umbrela_score_accepts_known_shapes(text, expected_grade, expected_valid):
    judge = UmbrelaJudge()
    grade, valid = judge._parse_umbrela_score(MinimaLlmResponse(request_id="x", text=text))
    assert grade == expected_grade
    assert valid == expected_valid


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I cannot judge this.",
        "four",
    ],
)
def test_parse_umbrela_score_returns_invalid_zero_on_malformed_output(text):
    judge = UmbrelaJudge()
    grade, valid = judge._parse_umbrela_score(MinimaLlmResponse(request_id="x", text=text))
    assert grade == 0.0
    assert valid is False


def test_parse_umbrela_score_returns_invalid_zero_for_non_response_error():
    judge = UmbrelaJudge()
    assert judge._parse_umbrela_score(NotAResponse()) == (0.0, False)
    assert judge._parse_umbrela_score(None) == (0.0, False)


def test_judge_grades_responses_and_builds_leaderboard(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    reports: List[Report] = [
        make_report("run1", "t1", "relevant response text"),
        make_report("run2", "t1", "irrelevant response text"),
    ]
    topics = [make_topic("t1", "some query")]

    async def fake_run_batched(self, requests):
        results = []
        for req in requests:
            if "run1" in str(req.request_id) or req.request_id == "q0_umbrela":
                text = "Reasoning...\n##final score: 3"
            else:
                text = "Reasoning...\n##final score: 0"
            results.append(MinimaLlmResponse(request_id=req.request_id, text=text))
        return results

    monkeypatch.setattr(
        "judges.umbrela_judge.umbrela_judge.OpenAIMinimaLlm.run_batched",
        fake_run_batched,
    )

    class FakeLlmConfig:
        raw = None

    leaderboard = UmbrelaJudge().judge(
        rag_responses=reports,
        rag_topics=topics,
        llm_config=FakeLlmConfig(),
    )

    by_run = {e.run_id: e for e in leaderboard.entries if e.topic_id == "t1"}
    assert by_run["run1"].values["UMBRELA_RELEVANCE"] == 3.0
    assert by_run["run1"].values["UMBRELA_RELEVANT"] == 1.0
    assert by_run["run2"].values["UMBRELA_RELEVANCE"] == 0.0
    assert by_run["run2"].values["UMBRELA_RELEVANT"] == 0.0
    assert by_run["run1"].values["UMBRELA_VALID_RESPONSE_RATE"] == 1.0


def test_create_qrels_produces_rows_with_run_id_as_doc_id(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fake")
    monkeypatch.setenv("OPENAI_MODEL", "fake-model")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    reports: List[Report] = [make_report("run1", "t1", "some text")]
    topics = [make_topic("t1", "some query")]

    async def fake_run_batched(self, requests):
        return [MinimaLlmResponse(request_id=r.request_id, text="##final score: 2") for r in requests]

    monkeypatch.setattr(
        "judges.umbrela_judge.umbrela_judge.OpenAIMinimaLlm.run_batched",
        fake_run_batched,
    )

    class FakeLlmConfig:
        raw = None

    qrels = UmbrelaJudge().create_qrels(
        rag_responses=reports,
        rag_topics=topics,
        llm_config=FakeLlmConfig(),
    )

    assert qrels is not None
    assert len(qrels.rows) == 1
    row = qrels.rows[0]
    assert row.topic_id == "t1"
    assert row.doc_id == "run1"
    assert row.grade == 2
