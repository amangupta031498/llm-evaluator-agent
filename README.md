# 📊 LLM Output Evaluator Agent

Uses the **Judge-LLM pattern** — each metric is a dedicated graph node with its own specialized system prompt.

---

## 🏗️ Architecture

```
input → faithfulness_node → relevance_node → hallucination_node → aggregator_node → report_node → END
```

Each node is an independent judge LLM call — this is a real multi-node LangGraph agent, not a single prompt.

### Metrics
| Metric | What it measures | Higher = |
|---|---|---|
| **Faithfulness** | Claims grounded in prompt/context? | Better ✅ |
| **Relevance** | Actually answers the question? | Better ✅ |
| **Hallucination Risk** | Confident unsupported assertions? | Worse ❌ |
| **Overall** | Weighted combination (hallucination inverted) | Better ✅ |

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a FREE Groq API key
- Sign up at https://console.groq.com (free, no credit card)
- Create an API key

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## 🚀 Run

```bash
python evaluator_agent.py
```

You'll be prompted to paste:
1. **The original prompt** — what was asked of the LLM
2. **The LLM response** — the output you want to evaluate
3. **Context** (optional) — any RAG chunks, documents, or grounding info

Type `END` on a new line to finish each input.

---

## 📄 Output

### Terminal
```
══════════════════════════════════════════════════════════════
  LLM OUTPUT EVALUATION REPORT
  2024-11-15 14:32:01
══════════════════════════════════════════════════════════════

  FAITHFULNESS  [████████░░] 8.0/10
  Score : 8.0/10
  Why   : The response stays close to the provided context...
  Fix   : Avoid the unsupported claim about X in paragraph 2.
  ──────────────────────────────────────────────────────────

  RELEVANCE     [█████████░] 9.0/10
  ...

  OVERALL SCORE : 8.3/10
  VERDICT       : GOOD — Reliable output with minor areas to improve.
══════════════════════════════════════════════════════════════
```

### JSON Report
Saved automatically to `eval_report.json` after every run.

---

## 💡 Use Cases
- Evaluate responses from any LLM (ChatGPT, Gemini, Claude, etc.)
- QA gate in a RAG pipeline before returning answers to users
- Compare prompt versions — run both, compare overall scores

---

## 🛠️ Tech Stack
- **LangGraph** — multi-node agent state machine
- **Groq** — free LLM inference (Llama 3.3 70B)
- **LangChain** — LLM abstraction layer
- **Python** — no UI dependencies needed

---


