"""
╔══════════════════════════════════════════════════════════════╗
║         LLM Output Evaluator Agent  |  LangGraph + Groq     ║
║   Metrics: Faithfulness · Relevance · Hallucination Risk     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import re
import sys
from datetime import datetime
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# ─────────────────────────────────────────────
#  STATE SCHEMA
# ─────────────────────────────────────────────

class EvalState(TypedDict):
    user_prompt: str
    llm_response: str
    context: str                          # optional grounding context

    faithfulness_score: Optional[float]
    faithfulness_reasoning: Optional[str]
    faithfulness_suggestion: Optional[str]

    relevance_score: Optional[float]
    relevance_reasoning: Optional[str]
    relevance_suggestion: Optional[str]

    hallucination_score: Optional[float]  # risk score: higher = more risky
    hallucination_reasoning: Optional[str]
    hallucination_suggestion: Optional[str]

    overall_score: Optional[float]
    verdict: Optional[str]
    error: Optional[str]


# ─────────────────────────────────────────────
#  LLM SETUP
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY not found in environment.")
        print("  1. Sign up free at https://console.groq.com")
        print("  2. Create an API key")
        print("  3. Add to .env file:  GROQ_API_KEY=your_key_here\n")
        sys.exit(1)
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        api_key=api_key,
    )


def call_judge(llm, system_prompt: str, user_content: str) -> dict:
    """Call the judge LLM and parse JSON response."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: extract JSON object from response
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from judge response:\n{raw}")


# ─────────────────────────────────────────────
#  NODE 1 — FAITHFULNESS JUDGE
# ─────────────────────────────────────────────

FAITHFULNESS_SYSTEM = """You are a strict Faithfulness Judge evaluating an LLM response.

FAITHFULNESS measures: Does the response only assert things that are supported by the provided context or prompt? 
A response is unfaithful if it introduces facts, claims, or conclusions not present or implied in the input.

Evaluate and return ONLY a JSON object with exactly these keys:
{
  "score": <float 0.0 to 10.0>,
  "reasoning": "<2-3 sentences explaining why this score was given>",
  "suggestion": "<one concrete, actionable improvement the LLM response could make>"
}

Scoring guide:
  9-10 = Fully grounded, every claim traceable to the prompt/context
  7-8  = Mostly faithful, minor extrapolations
  5-6  = Some claims go beyond the context
  3-4  = Significant unsupported assertions
  0-2  = Response largely fabricates or contradicts the input

Return ONLY valid JSON. No preamble, no explanation outside the JSON."""


def faithfulness_node(state: EvalState) -> EvalState:
    llm = get_llm()
    user_content = f"""ORIGINAL PROMPT:
{state['user_prompt']}

CONTEXT / GROUNDING INFORMATION:
{state['context'] or '(No additional context provided — evaluate against the prompt only)'}

LLM RESPONSE TO EVALUATE:
{state['llm_response']}

Evaluate the faithfulness of the LLM Response."""

    try:
        result = call_judge(llm, FAITHFULNESS_SYSTEM, user_content)
        return {
            **state,
            "faithfulness_score": float(result["score"]),
            "faithfulness_reasoning": result["reasoning"],
            "faithfulness_suggestion": result["suggestion"],
        }
    except Exception as e:
        return {**state, "error": f"Faithfulness node failed: {e}"}


# ─────────────────────────────────────────────
#  NODE 2 — RELEVANCE JUDGE
# ─────────────────────────────────────────────

RELEVANCE_SYSTEM = """You are a strict Relevance Judge evaluating an LLM response.

RELEVANCE measures: Does the response actually answer what the user asked? 
A response is irrelevant if it addresses a different question, goes off-topic, or buries the answer in padding.

Evaluate and return ONLY a JSON object with exactly these keys:
{
  "score": <float 0.0 to 10.0>,
  "reasoning": "<2-3 sentences explaining why this score was given>",
  "suggestion": "<one concrete, actionable improvement the LLM response could make>"
}

Scoring guide:
  9-10 = Directly and completely answers the question asked
  7-8  = Mostly on-target with minor off-topic sections
  5-6  = Partially relevant, significant tangents
  3-4  = Loosely related but misses the core question
  0-2  = Does not address the question at all

Return ONLY valid JSON. No preamble, no explanation outside the JSON."""


def relevance_node(state: EvalState) -> EvalState:
    if state.get("error"):
        return state
    llm = get_llm()
    user_content = f"""ORIGINAL PROMPT:
{state['user_prompt']}

LLM RESPONSE TO EVALUATE:
{state['llm_response']}

Evaluate how relevant the LLM Response is to the Original Prompt."""

    try:
        result = call_judge(llm, RELEVANCE_SYSTEM, user_content)
        return {
            **state,
            "relevance_score": float(result["score"]),
            "relevance_reasoning": result["reasoning"],
            "relevance_suggestion": result["suggestion"],
        }
    except Exception as e:
        return {**state, "error": f"Relevance node failed: {e}"}


# ─────────────────────────────────────────────
#  NODE 3 — HALLUCINATION RISK JUDGE
# ─────────────────────────────────────────────

HALLUCINATION_SYSTEM = """You are a strict Hallucination Risk Judge evaluating an LLM response.

HALLUCINATION RISK measures: Does the response contain specific claims (names, dates, numbers, URLs, events) 
that cannot be verified from the prompt/context, and that a reader might mistakenly treat as fact?
This is different from faithfulness — focus specifically on confident-sounding fabrications.

Evaluate and return ONLY a JSON object with exactly these keys:
{
  "score": <float 0.0 to 10.0>,
  "reasoning": "<2-3 sentences explaining why this score was given, citing specific examples if hallucinations found>",
  "suggestion": "<one concrete, actionable improvement the LLM response could make>"
}

NOTE: score here represents RISK LEVEL — higher score = higher hallucination risk (worse).
  0-2  = Very low risk, response is appropriately hedged and grounded
  3-4  = Low risk, minor unverifiable claims
  5-6  = Moderate risk, some confident unsupported assertions
  7-8  = High risk, multiple fabricated specifics
  9-10 = Very high risk, response is largely hallucinated

Return ONLY valid JSON. No preamble, no explanation outside the JSON."""


def hallucination_node(state: EvalState) -> EvalState:
    if state.get("error"):
        return state
    llm = get_llm()
    user_content = f"""ORIGINAL PROMPT:
{state['user_prompt']}

CONTEXT / GROUNDING INFORMATION:
{state['context'] or '(No additional context provided)'}

LLM RESPONSE TO EVALUATE:
{state['llm_response']}

Assess the hallucination risk of the LLM Response."""

    try:
        result = call_judge(llm, HALLUCINATION_SYSTEM, user_content)
        return {
            **state,
            "hallucination_score": float(result["score"]),
            "hallucination_reasoning": result["reasoning"],
            "hallucination_suggestion": result["suggestion"],
        }
    except Exception as e:
        return {**state, "error": f"Hallucination node failed: {e}"}


# ─────────────────────────────────────────────
#  NODE 4 — AGGREGATOR
# ─────────────────────────────────────────────

def aggregator_node(state: EvalState) -> EvalState:
    if state.get("error"):
        return state

    f = state["faithfulness_score"]
    r = state["relevance_score"]
    h = state["hallucination_score"]

    # Overall: average of faithfulness + relevance + inverted hallucination risk
    hallucination_quality = 10.0 - h  # invert: low risk = high quality
    overall = round((f + r + hallucination_quality) / 3, 2)

    if overall >= 8.5:
        verdict = "EXCELLENT — Production ready. High quality, trustworthy output."
    elif overall >= 7.0:
        verdict = "GOOD — Reliable output with minor areas to improve."
    elif overall >= 5.5:
        verdict = "FAIR — Usable but needs review before relying on this output."
    elif overall >= 4.0:
        verdict = "POOR — Significant issues found. Do not use without corrections."
    else:
        verdict = "FAIL — Output is unreliable. Regenerate with better prompting."

    return {**state, "overall_score": overall, "verdict": verdict}


# ─────────────────────────────────────────────
#  NODE 5 — REPORT (terminal + JSON)
# ─────────────────────────────────────────────

def score_bar(score: float, invert: bool = False) -> str:
    """ASCII progress bar for scores."""
    display = (10.0 - score) if invert else score
    filled = int(display)
    bar = "█" * filled + "░" * (10 - filled)
    return f"[{bar}] {score:.1f}/10"


def color_score(score: float, invert: bool = False) -> str:
    """ANSI color based on score quality."""
    quality = (10.0 - score) if invert else score
    if quality >= 8:
        return f"\033[92m{score:.1f}\033[0m"   # green
    elif quality >= 6:
        return f"\033[93m{score:.1f}\033[0m"   # yellow
    else:
        return f"\033[91m{score:.1f}\033[0m"   # red


def report_node(state: EvalState) -> EvalState:
    if state.get("error"):
        print(f"\n\033[91m[AGENT ERROR]\033[0m {state['error']}\n")
        return state

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "─" * 62

    print(f"\n\033[1m{'═' * 62}\033[0m")
    print(f"\033[1m  LLM OUTPUT EVALUATION REPORT\033[0m")
    print(f"  {ts}")
    print(f"\033[1m{'═' * 62}\033[0m\n")

    # ── FAITHFULNESS ──
    print(f"\033[1m  FAITHFULNESS\033[0m  {score_bar(state['faithfulness_score'])}")
    print(f"  Score : {color_score(state['faithfulness_score'])}/10")
    print(f"  Why   : {state['faithfulness_reasoning']}")
    print(f"  Fix   : \033[96m{state['faithfulness_suggestion']}\033[0m")
    print(f"  {sep}")

    # ── RELEVANCE ──
    print(f"\033[1m  RELEVANCE\033[0m     {score_bar(state['relevance_score'])}")
    print(f"  Score : {color_score(state['relevance_score'])}/10")
    print(f"  Why   : {state['relevance_reasoning']}")
    print(f"  Fix   : \033[96m{state['relevance_suggestion']}\033[0m")
    print(f"  {sep}")

    # ── HALLUCINATION ──
    print(f"\033[1m  HALLUCINATION\033[0m {score_bar(state['hallucination_score'], invert=True)}  (risk — lower is better)")
    print(f"  Risk  : {color_score(state['hallucination_score'], invert=True)}/10")
    print(f"  Why   : {state['hallucination_reasoning']}")
    print(f"  Fix   : \033[96m{state['hallucination_suggestion']}\033[0m")
    print(f"  {sep}")

    # ── OVERALL ──
    overall = state["overall_score"]
    overall_color = color_score(overall)
    print(f"\n\033[1m  OVERALL SCORE : {overall_color}/10\033[0m")
    print(f"  VERDICT       : \033[1m{state['verdict']}\033[0m")
    print(f"\n\033[1m{'═' * 62}\033[0m\n")

    # Save JSON report
    report = {
        "timestamp": ts,
        "input": {
            "prompt": state["user_prompt"],
            "response": state["llm_response"],
            "context": state["context"],
        },
        "metrics": {
            "faithfulness": {
                "score": state["faithfulness_score"],
                "reasoning": state["faithfulness_reasoning"],
                "suggestion": state["faithfulness_suggestion"],
            },
            "relevance": {
                "score": state["relevance_score"],
                "reasoning": state["relevance_reasoning"],
                "suggestion": state["relevance_suggestion"],
            },
            "hallucination_risk": {
                "score": state["hallucination_score"],
                "reasoning": state["hallucination_reasoning"],
                "suggestion": state["hallucination_suggestion"],
            },
        },
        "overall": {
            "score": overall,
            "verdict": state["verdict"],
        },
    }

    out_path = "eval_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  \033[90mReport saved → {out_path}\033[0m\n")

    return state


# ─────────────────────────────────────────────
#  BUILD THE LANGGRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(EvalState)

    graph.add_node("faithfulness", faithfulness_node)
    graph.add_node("relevance", relevance_node)
    graph.add_node("hallucination", hallucination_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("faithfulness")
    graph.add_edge("faithfulness", "relevance")
    graph.add_edge("relevance", "hallucination")
    graph.add_edge("hallucination", "aggregator")
    graph.add_edge("aggregator", "report")
    graph.add_edge("report", END)

    return graph.compile()


# ─────────────────────────────────────────────
#  CLI INTERFACE
# ─────────────────────────────────────────────

def get_multiline_input(prompt: str) -> str:
    """Read multiline input until user enters END on a blank line."""
    print(prompt)
    print("  (Type or paste your text. When done, type END on a new line and press Enter)\n")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def main():
    print("\n\033[1m╔══════════════════════════════════════════════════╗\033[0m")
    print("\033[1m║     LLM OUTPUT EVALUATOR AGENT  v1.0             ║\033[0m")
    print("\033[1m║     LangGraph · Groq · Judge-LLM Pattern         ║\033[0m")
    print("\033[1m╚══════════════════════════════════════════════════╝\033[0m\n")
    print("  Evaluates any LLM response on:")
    print("  • Faithfulness   • Relevance   • Hallucination Risk\n")

    # Collect inputs
    user_prompt = get_multiline_input("① ORIGINAL PROMPT (what was asked of the LLM):")
    print()

    llm_response = get_multiline_input("② LLM RESPONSE (the output you want to evaluate):")
    print()

    print("③ CONTEXT / GROUNDING INFO  (optional — paste docs, RAG chunks, etc.)")
    print("  Press Enter to skip, or provide context below:")
    print("  (Type END when done, or just END to skip)\n")
    context_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        context_lines.append(line)
    context = "\n".join(context_lines).strip()

    print("\n\033[90m  Running evaluation graph...\033[0m")
    print("\033[90m  faithfulness → relevance → hallucination → aggregator → report\033[0m\n")

    # Build and run the graph
    app = build_graph()
    initial_state: EvalState = {
        "user_prompt": user_prompt,
        "llm_response": llm_response,
        "context": context,
        "faithfulness_score": None,
        "faithfulness_reasoning": None,
        "faithfulness_suggestion": None,
        "relevance_score": None,
        "relevance_reasoning": None,
        "relevance_suggestion": None,
        "hallucination_score": None,
        "hallucination_reasoning": None,
        "hallucination_suggestion": None,
        "overall_score": None,
        "verdict": None,
        "error": None,
    }

    app.invoke(initial_state)


if __name__ == "__main__":
    main()
