from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.llm_runtime import _extract_json
from src.reflexion_lab.mock_runtime import MockRuntime
from src.reflexion_lab.reporting import build_report, failure_breakdown
from src.reflexion_lab.schemas import JudgeResult, ReflectionEntry
from src.reflexion_lab.utils import load_dataset


def test_schemas_have_required_fields():
    j = JudgeResult(score=1, reason="ok")
    assert j.score == 1 and j.missing_evidence == []
    r = ReflectionEntry(attempt_id=1, failure_reason="x", lesson="y", next_strategy="z")
    assert r.attempt_id == 1 and r.next_strategy == "z"


def test_extract_json_handles_fences_and_noise():
    assert _extract_json('```json\n{"score": 1, "reason": "a"}\n```')["score"] == 1
    assert _extract_json('here you go: {"score": 0, "reason": "b"} thanks')["score"] == 0
    assert _extract_json("not json") == {}


def test_reflexion_recovers_where_react_fails():
    data = load_dataset("data/hotpot_mini.json")
    rt = MockRuntime()
    react = [ReActAgent(runtime=rt).run(ex) for ex in data]
    reflexion = [ReflexionAgent(max_attempts=3, runtime=rt).run(ex) for ex in data]
    react_em = sum(r.is_correct for r in react) / len(react)
    reflexion_em = sum(r.is_correct for r in reflexion) / len(reflexion)
    assert reflexion_em > react_em
    # Reflexion should have produced at least one reflection entry on a failing item.
    assert any(r.reflections for r in reflexion)


def test_report_failure_modes_keyed_by_mode():
    data = load_dataset("data/hotpot_mini.json")
    rt = MockRuntime()
    records = [ReActAgent(runtime=rt).run(ex) for ex in data] + [ReflexionAgent(runtime=rt).run(ex) for ex in data]
    fb = failure_breakdown(records)
    # Each key is a failure-mode name with per-agent counts.
    assert "none" in fb
    assert all("total" in v for v in fb.values())
    report = build_report(records, "hotpot_mini.json", mode="mock")
    assert len(report.discussion) >= 250
    assert set(["meta", "summary", "failure_modes", "examples", "extensions", "discussion"]).issubset(report.model_dump().keys())
