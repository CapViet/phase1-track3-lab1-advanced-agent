from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class ContextChunk(BaseModel):
    title: str
    text: str

class QAExample(BaseModel):
    qid: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    gold_answer: str
    context: list[ContextChunk]

class JudgeResult(BaseModel):
    """Structured verdict produced by the Evaluator for a single answer."""
    score: int = Field(..., ge=0, le=1, description="1 if the answer is correct, 0 otherwise.")
    reason: str = Field(..., description="Short explanation of why the answer is correct or wrong.")
    missing_evidence: list[str] = Field(default_factory=list, description="Facts/hops the answer failed to ground or complete.")
    spurious_claims: list[str] = Field(default_factory=list, description="Claims in the answer not supported by the context.")

class ReflectionEntry(BaseModel):
    """A single self-reflection produced after a failed attempt."""
    attempt_id: int = Field(..., description="The attempt that this reflection analyses.")
    failure_reason: str = Field(..., description="Why the previous attempt was judged wrong.")
    lesson: str = Field(..., description="A generalisable lesson learned from the failure.")
    next_strategy: str = Field(..., description="Concrete strategy the Actor should try on the next attempt.")

class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0

class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)

class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str
    cost: dict = Field(default_factory=dict)

class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
