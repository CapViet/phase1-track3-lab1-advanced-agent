from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

from .mock_runtime import MockRuntime
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: object = field(default_factory=MockRuntime)

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        attempt_answers: list[str] = []
        final_answer = ""
        final_score = 0
        final_judge = None

        for attempt_id in range(1, self.max_attempts + 1):
            # --- Act: produce an answer (real token + latency come from the runtime) ---
            answer, act_usage = self.runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            # --- Evaluate: judge the answer against the gold answer ---
            judge, eval_usage = self.runtime.evaluator(example, answer)

            token_estimate = act_usage.total_tokens + eval_usage.total_tokens
            latency_ms = act_usage.latency_ms + eval_usage.latency_ms

            attempt_answers.append(answer)
            final_answer = answer
            final_score = judge.score
            final_judge = judge

            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)

            if judge.score == 1:
                traces.append(trace)
                break

            # --- Reflect: only for the Reflexion agent, and only if more attempts remain ---
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                entry, refl_usage = self.runtime.reflector(example, attempt_id, judge, answer)
                trace.reflection = entry
                trace.token_estimate += refl_usage.total_tokens
                trace.latency_ms += refl_usage.latency_ms
                reflections.append(entry)
                # Feed the new strategy back to the Actor for the next attempt.
                reflection_memory.append(entry.next_strategy)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        if final_score == 1:
            failure_mode = "none"
        else:
            failure_mode = self.runtime.classify_failure(example, final_answer, final_judge, attempt_answers)

        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)


class ReActAgent(BaseAgent):
    def __init__(self, runtime: Optional[object] = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime or MockRuntime())


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, runtime: Optional[object] = None) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, runtime=runtime or MockRuntime())
