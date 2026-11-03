"""
The 6 ORACLE agents. Each function maps to one agent role.
"""
from oracle.llm_clients import call_gpt, call_claude, call_gemini, call_deepseek, parse_json
from oracle.models import (
    Interpretation,
    AnalystResult,
    ClarifierResult,
    AdvocateCritique,
    JudgeVerdict,
    EvalScore,
)

# ── Agent 1 & 2: Interpreters ─────────────────────────────────────────────────

_INTERPRETER_SYSTEM = """You are an expert language interpreter.
Given a user query, you must:
1. Provide your best answer or intended meaning of the query.
2. State the context you assumed.
3. Rate your confidence from 0.0 to 1.0.

Respond ONLY with valid JSON:
{
  "answer": "...",
  "assumed_context": "...",
  "confidence": 0.XX
}"""


def interpreter_a(query: str) -> Interpretation:
    raw = call_gpt(_INTERPRETER_SYSTEM, query)
    data = parse_json(raw)
    return Interpretation(agent="Interpreter A (GPT-4o)", **data)


def interpreter_b(query: str) -> Interpretation:
    raw = call_claude(_INTERPRETER_SYSTEM, query)
    data = parse_json(raw)
    return Interpretation(agent="Interpreter B (Claude)", **data)


# ── Agent 3: Ambiguity Analyst ────────────────────────────────────────────────

_ANALYST_SYSTEM = """You are an expert in linguistic ambiguity classification.
You follow the AT-CoT taxonomy:
- Semantic: the same word/phrase has multiple meanings.
- Generalize: the query is too broad/vague.
- Specify: the query needs more specific details.
- None: the query is clear.

You receive two independent interpretations of the same query and decide:
1. Whether a conflict exists.
2. The ambiguity type.
3. An ambiguity score (0.0 = crystal clear, 1.0 = maximally ambiguous).
4. A brief reasoning chain.

Respond ONLY with valid JSON:
{
  "conflict_detected": true/false,
  "ambiguity_type": "Semantic|Generalize|Specify|None",
  "ambiguity_score": 0.XX,
  "reasoning": "..."
}"""


def ambiguity_analyst(query: str, interp_a: Interpretation, interp_b: Interpretation) -> AnalystResult:
    user_msg = f"""Query: {query}

Interpretation A ({interp_a.agent}):
  Answer: {interp_a.answer}
  Assumed context: {interp_a.assumed_context}
  Confidence: {interp_a.confidence}

Interpretation B ({interp_b.agent}):
  Answer: {interp_b.answer}
  Assumed context: {interp_b.assumed_context}
  Confidence: {interp_b.confidence}"""

    raw = call_gpt(_ANALYST_SYSTEM, user_msg)
    data = parse_json(raw)
    return AnalystResult(**data)


# ── Agent 4: Clarifier ────────────────────────────────────────────────────────

_CLARIFIER_SYSTEM = """You are an expert at asking precise clarifying questions.
Given a query and an ambiguity analysis, generate the single most important
clarifying question that would resolve the ambiguity.
Be concise and specific — one question only.

Respond ONLY with valid JSON:
{
  "question": "..."
}"""


def clarifier(query: str, analyst: AnalystResult, critique: str | None = None) -> str:
    user_msg = f"""Query: {query}

Ambiguity type: {analyst.ambiguity_type}
Ambiguity score: {analyst.ambiguity_score}
Reasoning: {analyst.reasoning}"""

    if critique:
        user_msg += f"\n\nDevil's Advocate critique of your previous question:\n{critique}\nRevise your question."

    raw = call_claude(_CLARIFIER_SYSTEM, user_msg)
    data = parse_json(raw)
    return data["question"]


# ── Agent 5: Devil's Advocate ─────────────────────────────────────────────────

_ADVOCATE_SYSTEM = """You are a Devil's Advocate reviewing a clarifying question.
Your job: challenge whether it is the *right* question. Attack the logic,
not just the wording. Consider: Does it address the core ambiguity? Is it
too broad? Does it assume something it shouldn't?

Respond ONLY with valid JSON:
{
  "critique": "...",
  "suggested_improvement": "..."
}"""


def devils_advocate(query: str, question_v1: str, analyst: AnalystResult) -> AdvocateCritique:
    user_msg = f"""Original query: {query}
Ambiguity analysis: {analyst.reasoning}
Clarifying question proposed: {question_v1}"""

    raw = call_gpt(_ADVOCATE_SYSTEM, user_msg)
    data = parse_json(raw)
    return AdvocateCritique(**data)


# ── Agent 6: Judge ────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """You are an expert judge evaluating clarifying questions.
Score each question on three axes (0.0–1.0 each):
- Relevance: does it directly address the ambiguity?
- Specificity: is it precise enough to be answerable?
- Necessity: would the answer actually change how the query is handled?

Pick a winner and explain briefly.

Respond ONLY with valid JSON:
{
  "relevance_v1": 0.XX,
  "specificity_v1": 0.XX,
  "necessity_v1": 0.XX,
  "relevance_v2": 0.XX,
  "specificity_v2": 0.XX,
  "necessity_v2": 0.XX,
  "winner": "v1" or "v2",
  "explanation": "..."
}"""


def judge(
    query: str,
    question_v1: str,
    question_v2: str,
    debate_transcript: str,
) -> JudgeVerdict:  # internal: picks winner between v1 and v2
    user_msg = f"""Query: {query}

Question v1: {question_v1}
Question v2: {question_v2}

Full debate transcript:
{debate_transcript}"""

    raw = call_claude(_JUDGE_SYSTEM, user_msg)
    data = parse_json(raw)
    return JudgeVerdict(**data)


# ── Method 1: Gemini Evaluation Judge ────────────────────────────────────────
# Exact prompt as specified in the scoring framework — do not modify.

_EVAL_JUDGE_PROMPT = """You are an expert evaluator of clarifying questions in NLP systems.
You are given an original query, the ambiguity type detected,
and a clarifying question generated by an AI system.

Score the clarifying question on these 3 axes from 1 to 5.
Also output two binary flags.

RELEVANCE (1-5): Does the question identify the correct ambiguity source?
    1 = completely wrong source
    2 = loosely related but misses the point
    3 = partially correct source
    4 = mostly correct with minor gaps
    5 = precisely identifies the ambiguity

SPECIFICITY (1-5): Does the question offer actionable disambiguation options?
    1 = "can you clarify?" (useless)
    2 = very broad, applies to any query
    3 = narrows domain but not specific interpretation
    4 = mostly specific, could be sharper
    5 = offers concrete mutually exclusive options

NECESSITY (1-5): Is clarification actually needed here?
    1 = query was already unambiguous (false positive)
    2 = clarification unhelpful
    3 = clarification helpful but not essential
    4 = clarification important
    5 = clarification essential for a correct answer

HALLUCINATION: Did the system answer confidently instead of asking? (0 or 1)
FALSE_POSITIVE: Did the system ask unnecessarily on a clear query? (0 or 1)

Respond ONLY in this exact JSON format, no extra text:
{{
  "relevance": <1-5>,
  "specificity": <1-5>,
  "necessity": <1-5>,
  "hallucination": <0 or 1>,
  "false_positive": <0 or 1>,
  "explanation": "<one sentence justifying your scores>"
}}

Query: {query}
Ambiguity Type: {ambiguity_type}
Clarifying Question: {clarifying_question}"""


def eval_judge(
    query: str,
    ambiguity_type: str,
    clarifying_question: str,
) -> EvalScore:
    prompt = _EVAL_JUDGE_PROMPT.format(
        query=query,
        ambiguity_type=ambiguity_type,
        clarifying_question=clarifying_question if clarifying_question
                            else "None — the system answered directly without asking a clarifying question.",
    )
    raw = call_gemini("", prompt)
    data = parse_json(raw)
    return EvalScore(**data)


def deepseek_eval_judge(
    query: str,
    ambiguity_type: str,
    clarifying_question: str,
) -> EvalScore:
    prompt = _EVAL_JUDGE_PROMPT.format(
        query=query,
        ambiguity_type=ambiguity_type,
        clarifying_question=clarifying_question if clarifying_question
                            else "None — the system answered directly without asking a clarifying question.",
    )
    raw = call_deepseek("You are an expert NLP evaluator. Respond only with valid JSON.", prompt)
    data = parse_json(raw)
    return EvalScore(**data)
