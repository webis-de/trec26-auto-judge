from types import SimpleNamespace

import pytest

from judges.auto_nuggetizer import auto_nuggetizer


def make_response(team_id="team", run_id="run"):
    return SimpleNamespace(
        metadata=SimpleNamespace(team_id=team_id, run_id=run_id),
        get_report_text=lambda: "the response text",
    )


def make_nuggets():
    return [
        SimpleNamespace(question="vital nugget 1", importance="vital"),
        SimpleNamespace(question="vital nugget 2", importance="vital"),
        SimpleNamespace(question="okay nugget 1", importance="okay"),
        SimpleNamespace(question="okay nugget 2", importance="okay"),
    ]


def test_official_measures_are_declared_in_spec():
    measure_names = [m.name for m in auto_nuggetizer.AUTO_NUGGETIZER_SPEC.measures]

    for expected in (
        "strict_vital_score",
        "strict_all_score",
        "vital_score",
        "all_score",
    ):
        assert expected in measure_names


def test_assign_nuggets_reports_official_nuggetizer_scores(monkeypatch, tmp_path):
    assignments = {
        "vital nugget 1": "support",
        "vital nugget 2": "partial_support",
        "okay nugget 1": "support",
        "okay nugget 2": "not_support",
    }

    class FakeNuggetizer:
        def __init__(self, *args, **kwargs):
            pass

        def assign(self, query, response_text, nuggets):
            return [
                SimpleNamespace(
                    text=nugget.text,
                    importance=nugget.importance,
                    assignment=assignments[nugget.text],
                )
                for nugget in nuggets
            ]

    monkeypatch.setattr(
        "nuggetizer.models.nuggetizer.Nuggetizer", FakeNuggetizer
    )

    out_file = tmp_path / "assignments.jsonl"
    llm_config = SimpleNamespace(model="fake-model", api_key="key", base_url="http://fake")

    result = auto_nuggetizer.AutoNuggetizer().assign_nuggets(
        topic_id="topic-1",
        query="what is the query",
        response=make_response(),
        nuggets=make_nuggets(),
        llm_config=llm_config,
        out=str(out_file),
    )

    # 2 vital nuggets: 1 support, 1 partial_support
    assert result["strict_vital_score"] == pytest.approx(0.5)
    assert result["vital_score"] == pytest.approx(0.75)

    # 4 nuggets total: 2 support, 1 partial_support, 1 not_support
    assert result["strict_all_score"] == pytest.approx(0.5)
    assert result["all_score"] == pytest.approx(0.625)

    assert out_file.exists()
