"""Deterministic mock runtime.

Kept from the scaffold so the benchmark can run for free and produce reproducible output
(this is the `mock_mode_for_autograding` extension). It mimics an LLM that stops after the
first hop on a few questions and recovers once a reflection is available. The same three
roles are exposed by `LLMRuntime` in `llm_runtime.py`, backed by a real model.
"""
from __future__ import annotations

from .llm import Usage
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> str:
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid]
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid]
    return example.gold_answer


def evaluator(example: QAExample, answer: str) -> JudgeResult:
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization.")
    if normalize_answer(answer) == "london":
        return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[])
    return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer])


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> ReflectionEntry:
    strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
    return ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)


def _estimate(base: int, attempt_id: int, agent_type: str, extra: int) -> int:
    return base + (attempt_id * 65) + (120 if agent_type == "reflexion" else 0) + extra


class MockRuntime:
    """Wraps the deterministic functions above and emits estimated usage."""

    name = "mock"

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, Usage]:
        text = actor_answer(example, attempt_id, agent_type, reflection_memory)
        usage = Usage(prompt_tokens=_estimate(220, attempt_id, agent_type, 0), completion_tokens=40, latency_ms=160 + attempt_id * 40 + (90 if agent_type == "reflexion" else 0))
        return text, usage

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, Usage]:
        judge = evaluator(example, answer)
        return judge, Usage(prompt_tokens=120, completion_tokens=40, latency_ms=70)

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult, predicted: str) -> tuple[ReflectionEntry, Usage]:
        entry = reflector(example, attempt_id, judge)
        return entry, Usage(prompt_tokens=150, completion_tokens=60, latency_ms=110)

    def classify_failure(self, example: QAExample, predicted: str, judge: JudgeResult, attempt_answers: list[str]) -> str:
        return FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")
