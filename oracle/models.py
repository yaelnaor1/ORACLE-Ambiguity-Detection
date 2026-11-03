from pydantic import BaseModel, Field
from typing import Literal


class Interpretation(BaseModel):
    agent: str
    answer: str
    assumed_context: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalystResult(BaseModel):
    ambiguity_type: Literal["Semantic", "Generalize", "Specify", "None"]
    ambiguity_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    conflict_detected: bool


class ClarifierResult(BaseModel):
    question_v1: str
    question_v2: str | None = None


class AdvocateCritique(BaseModel):
    critique: str
    suggested_improvement: str


# ── Internal ORACLE judge: picks winner between v1 and v2 ────────────────────

class JudgeVerdict(BaseModel):
    relevance_v1: float = Field(ge=0.0, le=1.0)
    specificity_v1: float = Field(ge=0.0, le=1.0)
    necessity_v1: float = Field(ge=0.0, le=1.0)
    relevance_v2: float = Field(ge=0.0, le=1.0)
    specificity_v2: float = Field(ge=0.0, le=1.0)
    necessity_v2: float = Field(ge=0.0, le=1.0)
    winner: Literal["v1", "v2"]
    explanation: str


# ── Method 1: Gemini eval judge scores (3 axes, kept separate — never collapsed) ──

class EvalScore(BaseModel):
    relevance: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    necessity: int = Field(ge=1, le=5)
    hallucination: int = Field(ge=0, le=1)   # 1 = answered confidently on ambiguous query
    false_positive: int = Field(ge=0, le=1)  # 1 = asked clarification on clear query
    explanation: str


# ── Method 2: Automatic reference-based scores ────────────────────────────────

class AutoScore(BaseModel):
    rouge_l: float = Field(ge=0.0, le=1.0)
    bertscore: float  # rescaled BERTScore can be slightly negative
    arr: bool | None = None  # None = not applicable (clear query or no gold answer)
    exact_match: bool | None = None       # strict normalized question match
    nearby_match: bool | None = None      # semantic question similarity >= 0.85
    answer_match: str | None = None       # "exact" | "close" | "far" | None


class OracleResult(BaseModel):
    query: str
    is_ambiguous: bool
    interpretation_a: Interpretation
    interpretation_b: Interpretation
    analyst: AnalystResult
    clarifier: ClarifierResult | None = None
    advocate: AdvocateCritique | None = None
    judge: JudgeVerdict | None = None
    final_question: str | None = None
