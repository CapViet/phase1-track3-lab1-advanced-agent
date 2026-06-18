"""Real LLM runtime: the three Reflexion roles backed by an actual model.

This is the production replacement for `mock_runtime.py`. Each role builds the appropriate
prompt, calls the model through `LLMClient`, parses the response, and returns the parsed
object together with the real `Usage` (token counts + measured latency) reported by the
provider.
"""
from __future__ import annotations

import json
import re

from .llm import LLMClient, Usage
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


def _format_context(example: QAExample) -> str:
    return "\n".join(f"[{c.title}] {c.text}" for c in example.context)


def _extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


class LLMRuntime:
    """Drop-in replacement for MockRuntime, powered by a real model."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self.name = f"llm:{self.client.provider}:{self.client.model}"

    # ---- Actor ----------------------------------------------------------
    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, Usage]:
        parts = [f"QUESTION:\n{example.question}", f"\nCONTEXT:\n{_format_context(example)}"]
        if reflection_memory:
            notes = "\n".join(f"- {note}" for note in reflection_memory)
            parts.append(f"\nREFLECTION NOTES FROM PREVIOUS ATTEMPTS (apply these):\n{notes}")
        parts.append("\nFinal answer:")
        resp = self.client.chat(ACTOR_SYSTEM, "\n".join(parts))
        return self._clean_answer(resp.text), resp.usage

    @staticmethod
    def _clean_answer(text: str) -> str:
        text = text.strip()
        # Strip common chatty prefixes the model may add despite instructions.
        text = re.sub(r"(?i)^(the\s+)?(final\s+)?answer\s*[:\-]\s*", "", text).strip()
        # Keep only the first line.
        text = text.splitlines()[0].strip() if text else text
        return text.strip().strip('"').strip()

    # ---- Evaluator ------------------------------------------------------
    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, Usage]:
        user = (
            f"QUESTION:\n{example.question}\n\n"
            f"GOLD_ANSWER:\n{example.gold_answer}\n\n"
            f"PREDICTED_ANSWER:\n{answer}\n"
        )
        resp = self.client.chat(EVALUATOR_SYSTEM, user)
        data = _extract_json(resp.text)
        judge = self._parse_judge(data, example, answer)
        return judge, resp.usage

    @staticmethod
    def _parse_judge(data: dict, example: QAExample, answer: str) -> JudgeResult:
        # Trust the model's verdict but fall back to exact-match if parsing failed.
        if "score" in data:
            try:
                return JudgeResult(
                    score=1 if int(data.get("score", 0)) == 1 else 0,
                    reason=str(data.get("reason", "")) or "No reason provided.",
                    missing_evidence=[str(x) for x in data.get("missing_evidence", []) if x],
                    spurious_claims=[str(x) for x in data.get("spurious_claims", []) if x],
                )
            except (TypeError, ValueError):
                pass
        exact = normalize_answer(example.gold_answer) == normalize_answer(answer)
        return JudgeResult(score=1 if exact else 0, reason="Fell back to normalized exact-match grading.")

    # ---- Reflector ------------------------------------------------------
    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult, predicted: str) -> tuple[ReflectionEntry, Usage]:
        feedback = judge.reason
        if judge.missing_evidence:
            feedback += " Missing: " + "; ".join(judge.missing_evidence)
        if judge.spurious_claims:
            feedback += " Unsupported: " + "; ".join(judge.spurious_claims)
        user = (
            f"QUESTION:\n{example.question}\n\n"
            f"CONTEXT:\n{_format_context(example)}\n\n"
            f"PREDICTED_ANSWER (wrong, attempt {attempt_id}):\n{predicted}\n\n"
            f"JUDGE_FEEDBACK:\n{feedback}\n"
        )
        resp = self.client.chat(REFLECTOR_SYSTEM, user)
        data = _extract_json(resp.text)
        entry = ReflectionEntry(
            attempt_id=int(data.get("attempt_id", attempt_id) or attempt_id),
            failure_reason=str(data.get("failure_reason", "")) or feedback,
            lesson=str(data.get("lesson", "")) or "Complete every hop and ground each step in the context.",
            next_strategy=str(data.get("next_strategy", "")) or "Re-read the context and resolve each hop explicitly before answering.",
        )
        return entry, resp.usage

    # ---- Failure-mode classification -----------------------------------
    def classify_failure(self, example: QAExample, predicted: str, judge: JudgeResult, attempt_answers: list[str]) -> str:
        norm = normalize_answer(predicted)
        # Repeated identical wrong answers across attempts => stuck in a loop.
        wrong = [a for a in attempt_answers]
        if len(wrong) >= 2 and len({normalize_answer(a) for a in wrong}) == 1:
            return "looping"
        if not norm or norm in {"i dont know", "idk", "unknown", "none"}:
            return "incomplete_multi_hop"
        if judge.missing_evidence:
            return "incomplete_multi_hop"
        if judge.spurious_claims:
            return "entity_drift"
        return "wrong_final_answer"


def get_runtime(mode: str, **kwargs):
    """Factory: return the runtime for the given mode ('mock' or 'real'/'llm')."""
    if mode in ("mock",):
        from .mock_runtime import MockRuntime

        return MockRuntime()
    if mode in ("real", "llm"):
        return LLMRuntime(**kwargs)
    raise ValueError(f"Unknown runtime mode: {mode!r}")
