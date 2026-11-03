"""
Test the full pipeline on 1 query before committing to 500.
"""
from data.dataset_loader import load_csv
from evaluation.runner import _run_one

row = load_csv()[0]
print(f"Query        : {row['query']}")
print(f"Ground truth : ambiguous={row['is_ambiguous']}")
print(f"Gold question: {row['gold_question']}")
print(f"Gold answer  : {row['gold_answer']}")
print()

result = _run_one(row)


def show_scores(prefix: str, label: str):
    print(f"  [{label}]")
    print(f"    Relevance    : {result[f'{prefix}_relevance']}/5")
    print(f"    Specificity  : {result[f'{prefix}_specificity']}/5")
    print(f"    Necessity    : {result[f'{prefix}_necessity']}/5")
    print(f"    Hallucination: {result[f'{prefix}_hallucination']}")
    print(f"    False Pos    : {result[f'{prefix}_false_positive']}")
    print(f"    Explanation  : {result[f'{prefix}_explanation']}")
    print(f"    ROUGE-L      : {result[f'{prefix}_rouge_l']}")
    print(f"    BERTScore    : {result[f'{prefix}_bertscore']}")
    print(f"    ARR          : {result[f'{prefix}_arr']}")


print("=== BASELINE GPT ===")
print(f"  Response: {result['baseline_gpt_response']}")
show_scores("baseline_gpt", "EVAL JUDGE + AUTO")
print()

print("=== BASELINE CLAUDE ===")
print(f"  Response: {result['baseline_claude_response']}")
show_scores("baseline_claude", "EVAL JUDGE + AUTO")
print()

print("=== DEBATE-LITE ===")
print(f"  Ambiguous : {result['dlite_is_ambiguous']}")
print(f"  Type      : {result['dlite_ambiguity_type']}")
print(f"  Question  : {result['dlite_final_question']}")
show_scores("dlite", "EVAL JUDGE + AUTO")
print()

print("=== ORACLE ===")
print(f"  Ambiguous    : {result['oracle_is_ambiguous']}")
print(f"  Type         : {result['oracle_ambiguity_type']}")
print(f"  Question v1  : {result['oracle_question_v1']}")
print(f"  Question v2  : {result['oracle_question_v2']}")
print(f"  Winner       : {result['oracle_winner']}")
print(f"  Final Q      : {result['oracle_final_question']}")
show_scores("oracle", "EVAL JUDGE + AUTO")
