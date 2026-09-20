from types import SimpleNamespace
import json

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


def test_assign_nuggets_treats_unexpected_assignment_value_as_not_support(monkeypatch, tmp_path):
    """Covers the "Unexpected assignment: positive_effects" failure mode.

    The real Nuggetizer.assign() call can, for a malformed or unusual LLM
    response, return an assignment label outside the expected
    ("partial_support", "support", "not_support", "failed") set. Rather than
    retrying the call or crashing the whole judging run, assign_nuggets
    should immediately treat such labels as "not_support".
    """

    call_count = {"n": 0}

    class FakeNuggetizer:
        def __init__(self, *args, **kwargs):
            pass

        def assign(self, query, response_text, nuggets):
            call_count["n"] += 1
            return [
                SimpleNamespace(
                    text=nugget.text,
                    importance=nugget.importance,
                    # Simulate an LLM response that yields an assignment
                    # label the code does not know how to handle.
                    assignment="positive_effects",
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

    # No retry should happen for unexpected assignment values (only for
    # actual call failures), so assign() is called exactly once.
    assert call_count["n"] == 1

    # All nuggets fall back to "not_support", so nothing is credited as
    # supported and no exception is raised.
    assert result["vital_and_okay"] == 0
    assert result["strict_all_score"] == pytest.approx(0.0)
    assert result["all_score"] == pytest.approx(0.0)

    assert out_file.exists()
    logged = [json.loads(line) for line in out_file.read_text().splitlines()]
    # The raw (unexpected) label is still recorded in the output log for
    # debugging purposes; only the in-memory metrics treat it as not_support.
    assert all(entry["assignment"] == "positive_effects" for entry in logged)


def test_assign_nuggets_treats_unexpected_importance_value_as_okay(monkeypatch, tmp_path):
    """An unrecognized importance label (e.g. "important") must not crash
    the run. It should be treated as "okay" (the non-vital default), matching
    how nuggetizer.core.metrics.calculate_nugget_scores only special-cases
    "vital" and buckets everything else into "all".
    """

    class FakeNuggetizer:
        def __init__(self, *args, **kwargs):
            pass

        def assign(self, query, response_text, nuggets):
            return [
                SimpleNamespace(text=nugget.text, importance="important", assignment="support")
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

    # No exception is raised; all 4 nuggets are counted as non-vital ("okay"),
    # so the official (correct) all_score credits them and strict_vital_score
    # is unaffected since there are no vital nuggets.
    assert result["vital_and_okay"] == 4
    assert result["strict_vital_score"] == pytest.approx(0.0)
    assert result["strict_all_score"] == pytest.approx(1.0)

    logged = [json.loads(line) for line in out_file.read_text().splitlines()]
    # The raw (unexpected) label is still recorded in the output log.
    assert all(entry["importance"] == "important" for entry in logged)
