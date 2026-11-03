# ORACLE — Multi-Agent Ambiguity Detection & Clarification

A research project comparing three systems for detecting ambiguous queries and generating clarifying questions.

> Team final project, B.Sc. Industrial Engineering & Management, Ben-Gurion University. Copied with permission from the original repository by [Talya Saada](https://github.com/Talyasaa), co-author.

## Systems Compared

| System | Description | API Calls |
|---|---|---|
| Baseline GPT / Claude | Single model, direct answer | 1 |
| Debate-Lite | 2 interpreters + analyst + clarifier | 4 |
| **ORACLE** | Full 6-agent pipeline | ~9 |

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Talyasaa/ORACLE-Ambiguity-Detection.git
cd ORACLE-Ambiguity-Detection
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API keys

Copy the example file and fill in your keys:
```bash
cp .env.example .env
```

Then open `.env` and add:

```
OPENAI_API_KEY=sk-...           # from platform.openai.com
ANTHROPIC_API_KEY=sk-ant-...    # from console.anthropic.com
GEMINI_API_KEY=AIza...          # from aistudio.google.com
HUGGINGFACE_TOKEN=hf_...        # from huggingface.co/settings/tokens
DEEPSEEK_API_KEY=sk-...         # optional — only for the second judge (ICC). from platform.deepseek.com
```

### 5. Download the dataset
```bash
python -m data.dataset_loader
```
This downloads 500 queries from AmbigQA (333 ambiguous + 167 clear) and saves them to `data/ambig_qa.csv`.

---

## Running the Project

### Test on 1 query (before running all 500)
```bash
python -m evaluation.test_single
```

### Run the full evaluation (all 500 queries, all 3 systems)
```bash
python -m evaluation.runner
```
- Safe to stop and resume — saves row by row
- Already-completed rows are skipped automatically

### See the results table
```bash
python -m evaluation.aggregate
```

### Test a single query interactively
```bash
python -m main
```

### Additional analyses (after the main run)
```bash
# Ambiguity-detection metrics: precision/recall/F1, ROC/PR-AUC, threshold sweep
python -m evaluation.detection

# Detection plots (saved to evaluation/plots/)
python -m evaluation.visualize confusion   # confusion matrices
python -m evaluation.visualize f1curve     # F1 vs threshold
python -m evaluation.visualize detection   # P/R/F1 bar chart

# Second judge (DeepSeek) + inter-judge agreement (ICC)
python -m evaluation.rescore_deepseek          # adds DeepSeek scores -> results_with_deepseek.csv
python -m evaluation.icc gemini_vs_deepseek    # ICC: Gemini vs DeepSeek
```

---

## Reasoning Files — Understanding the Decisions

`results.csv` stores each system's *output* (clarifying question, scores) but not
*why* it produced that output. To inspect the full decision trail — each
interpreter's read on the query, the analyst's ambiguity diagnosis, the
devil's-advocate critique, and the judge's verdict — generate per-query
Markdown transcripts:

```bash
python -m evaluation.reasoning_files            # 5 sample queries (cheap preview)
python -m evaluation.reasoning_files --n 20     # first 20 (10 ambiguous + 10 clear)
```

Saves one `.md` file per query to `evaluation/reasoning_files/`. **Note:** this
re-runs the live pipeline (the reasoning was never persisted during the original
run), so each query costs ~10 LLM calls (ORACLE + Debate-Lite combined) — keep
`--n` small unless you need the full set.

---

## Project Structure

```
ORACLE-Ambiguity-Detection/
├── oracle/
│   ├── agents.py          # 6 ORACLE agents + Gemini & DeepSeek eval judges
│   ├── pipeline.py        # ORACLE full pipeline
│   ├── debate_lite.py     # Debate-Lite pipeline
│   ├── baseline.py        # Baseline GPT and Claude
│   ├── llm_clients.py     # OpenAI / Anthropic / Gemini / DeepSeek clients
│   ├── models.py          # Pydantic data models
│   └── config.py          # API keys and model config
├── data/
│   ├── dataset_loader.py  # Downloads AmbigQA from HuggingFace
│   └── ambig_qa.csv       # Generated — not in repo (gitignored)
├── evaluation/
│   ├── runner.py                  # Runs all queries through all systems (Method 1 + 2)
│   ├── auto_scorer.py             # ROUGE-L, BERTScore, ARR, answer-match
│   ├── detection.py               # Detection metrics: precision/recall/F1, AUC, sweep
│   ├── rescore_deepseek.py        # Adds DeepSeek as a second judge
│   ├── rescore_answer_quality.py  # Adds exact/near/answer-match columns
│   ├── icc.py                     # Inter-judge agreement (Gemini vs DeepSeek)
│   ├── visualize.py               # Detection plots
│   ├── reasoning_files.py         # Per-query decision transcripts
│   ├── aggregate.py               # Prints final results tables
│   ├── test_single.py             # Quick single-query test
│   ├── plots/                     # Generated detection figures
│   ├── reasoning_files/           # Generated per-query transcripts
│   ├── results.csv                # Main results (committed)
│   ├── results_with_deepseek.csv  # + DeepSeek judge scores (committed)
│   └── results_enriched.csv       # + exact/near/answer-match (committed)
├── docs/
│   ├── report.tex         # Final report (LaTeX)
│   └── ARR.md             # ARR metric walkthrough
├── main.py                # Interactive single-query demo
├── requirements.txt
└── .env.example           # Template for API keys
```

---

## API Keys You Need

| Key | Where to get it | Used for |
|---|---|---|
| `OPENAI_API_KEY` | platform.openai.com | GPT-4o (pipeline + ARR scoring) |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Claude (pipeline) |
| `GEMINI_API_KEY` | aistudio.google.com → Get API Key | Gemini (primary independent eval judge) |
| `HUGGINGFACE_TOKEN` | huggingface.co → Settings → Tokens | Download AmbigQA dataset |
| `DEEPSEEK_API_KEY` *(optional)* | platform.deepseek.com | DeepSeek (second eval judge, for ICC) |

> **Note:** Gemini (the project uses `gemini-2.0-flash`) requires billing enabled on Google Cloud — the free-tier quota for flash models can be 0. `DEEPSEEK_API_KEY` is optional and only needed for the second-judge / ICC analysis.

---

## Scoring Methods

| Method | What it measures |
|---|---|
| Gemini Judge | Relevance / Specificity / Necessity (1–5) + Hallucination / False Positive flags |
| DeepSeek Judge | Second independent judge; agreement with Gemini measured via ICC (`evaluation/icc.py`) |
| ROUGE-L | Word overlap between generated and gold clarifying question |
| BERTScore | Semantic similarity between generated and gold clarifying question |
| ARR | Did the clarification actually lead to the correct answer? See [`docs/ARR.md`](docs/ARR.md) for the full calculation walkthrough + worked examples |
| Detection metrics | Precision / Recall / F1 / Accuracy + ROC-AUC / PR-AUC for the ambiguity ask-or-answer decision (`evaluation/detection.py`) |
