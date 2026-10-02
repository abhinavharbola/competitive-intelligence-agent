import pytest
from pydantic import ValidationError

from agent import schemas


def test_valid_planner_steps_keeps_valid_steps():
    raw = {"steps": [
        {"sub_question": "a", "field": "risks", "tool": "search"},
        {"sub_question": "b", "field": "competitors", "tool": "calculator"},
    ]}
    steps = schemas.valid_planner_steps(raw)
    assert [s.field for s in steps] == ["risks", "competitors"]


def test_valid_planner_steps_drops_unknown_field():
    raw = {"steps": [
        {"sub_question": "a", "field": "not_a_field", "tool": "search"},
        {"sub_question": "b", "field": "risks", "tool": "search"},
    ]}
    steps = schemas.valid_planner_steps(raw)
    assert len(steps) == 1
    assert steps[0].field == "risks"


def test_valid_planner_steps_all_invalid_returns_empty():
    raw = {"steps": [{"sub_question": "a", "field": "x", "tool": "search"}]}
    assert schemas.valid_planner_steps(raw) == []


def test_valid_planner_steps_rejects_bad_tool():
    raw = {"steps": [{"sub_question": "a", "field": "risks", "tool": "browser"}]}
    with pytest.raises(ValidationError):
        schemas.valid_planner_steps(raw)


def test_valid_planner_steps_rejects_missing_steps_key():
    with pytest.raises(ValidationError):
        schemas.valid_planner_steps({})


def test_critic_response_requires_gaps():
    with pytest.raises(ValidationError):
        schemas.CriticResponse.model_validate({"approved": True})


def test_critic_response_approved_is_optional():
    parsed = schemas.CriticResponse.model_validate({"gaps": ["risks"]})
    assert parsed.approved is None and parsed.gaps == ["risks"]


def test_executor_args_defaults_to_empty_strings():
    args = schemas.ExecutorArgs.model_validate({})
    assert args.query == "" and args.expression == ""


def test_synthesizer_response_tolerates_unknown_status_value():
    parsed = schemas.SynthesizerResponse.model_validate(
        {"report_markdown": "x", "field_status": {"risks": "maybe"}}
    )
    assert parsed.field_status == {"risks": "maybe"}


def test_synthesizer_response_requires_report():
    with pytest.raises(ValidationError):
        schemas.SynthesizerResponse.model_validate({"field_status": {}})


def test_clean_field_status_coerces_drops_and_fills():
    import config
    cleaned = schemas.clean_field_status(
        {"risks": "confirmed", "competitors": "maybe", "bogus": "confirmed"}
    )
    assert set(cleaned) == set(config.REQUIRED_FIELDS)
    assert cleaned["risks"] == "confirmed"
    assert cleaned["competitors"] == "insufficient information"
    assert cleaned["what_it_does"] == "insufficient information"


def test_synthesizer_response_accepts_valid_payload():
    parsed = schemas.SynthesizerResponse.model_validate(
        {"report_markdown": "x", "field_status": {"risks": "confirmed"}}
    )
    assert parsed.field_status["risks"] == "confirmed"


