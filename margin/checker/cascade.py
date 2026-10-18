"""Checker cascade: SymPy (exact) -> Nemotron Nano (screen, recall) -> Nemotron Super (confirm, precision).

Nothing is flagged to the student unless SymPy can't vouch for the step AND Super confirms it.
"""

import json
import time
from dataclasses import dataclass, field

from margin.checker.algebra import StepCheck, Verdict, check_step
from margin.nebius import Nebius


@dataclass
class WorkLine:
    text: str
    sympy: str | None = None


@dataclass
class Flag:
    line: int  # 1-based
    reason: str
    confidence: float
    hint: str = ""


@dataclass
class CheckReport:
    flags: list[Flag]
    algebra: dict[int, StepCheck]
    screened: list[int]
    timings: dict[str, float] = field(default_factory=dict)
    calls: int = 0


SCREEN_PROMPT = """You are checking a student's handwritten work step by step. Course: {course}.
Question: {question}

Work (numbered lines):
{lines}

Exact algebra checks already done by a computer algebra system:
{algebra}

For EVERY line, decide whether it validly follows from the question and the previous lines.
Return JSON: {{"lines": [{{"line": n, "status": "ok" | "error" | "unsure" | "incomplete", "confidence": 0-1, "reason": "short"}}]}}
Mark a line "error" only for a real mistake (wrong formula, sign, substitution, arithmetic, logic), not style or skipped steps.
Lines marked equivalent by the algebra system are correct algebra; trust that."""

CONFIRM_PROMPT = """A screening pass flagged possible mistakes in a student's work. Course: {course}.
Question: {question}

Work (numbered lines):
{lines}

Candidate mistakes:
{candidates}

For each candidate, think carefully and decide whether it is REALLY a mistake. False alarms are costly:
a student who is told a correct step is wrong loses trust. If the line is correct, or only differs in
notation/rounding, it is NOT a mistake.
Return JSON: {{"verdicts": [{{"line": n, "is_mistake": true|false, "confidence": 0-1, "reason": "short", "hint": "one Socratic question that nudges without giving the answer"}}]}}"""


def check(
    client: Nebius | None,
    question: str,
    work: list[WorkLine],
    course: str = "unknown",
    screen_threshold: float = 0.5,
    confirm_threshold: float = 0.7,
) -> CheckReport:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    sympy_lines = link_continuations([w.sympy for w in work])
    algebra = {i + 1: check_step(sympy_lines[i - 1], sympy_lines[i]) for i in range(1, len(work))}
    timings["algebra_s"] = time.perf_counter() - t0

    if client is None:
        flags = [
            Flag(n, f"algebra: {c.reason}", 0.9)
            for n, c in algebra.items()
            if c.verdict == Verdict.NOT_EQUIVALENT
        ]
        return CheckReport(flags, algebra, [], timings)

    numbered = "\n".join(f"{i}. {w.text}" for i, w in enumerate(work, 1))
    algebra_text = "\n".join(f"line {n}: {c.verdict.value} ({c.reason})" for n, c in algebra.items()) or "none"
    calls = 0

    # 1. Screen with the fast model.
    screen = client.chat_json(
        client.cfg.model_fast,
        [{"role": "user", "content": SCREEN_PROMPT.format(course=course, question=question, lines=numbered, algebra=algebra_text)}],
        max_tokens=800,
    )
    calls += 1
    timings["screen_s"] = screen.latency_s
    candidates: dict[int, str] = {}
    for item in (screen.data or {}).get("lines") or []:
        try:
            n = int(item.get("line"))
        except (TypeError, ValueError):
            continue
        if algebra.get(n) and algebra[n].verdict == Verdict.EQUIVALENT:
            continue  # exact algebra overrides the model
        if item.get("status") == "error" and float(item.get("confidence") or 0) >= screen_threshold:
            candidates[n] = str(item.get("reason") or "")
    for n, c in algebra.items():
        if c.verdict == Verdict.NOT_EQUIVALENT:
            candidates.setdefault(n, f"algebra system: {c.reason}")

    if not candidates:
        return CheckReport([], algebra, [], timings, calls)

    # 2. Confirm with the strong model (reasoning on: this is the precision step).
    cand_text = "\n".join(f"line {n}: {reason}" for n, reason in sorted(candidates.items()))
    confirm = client.chat_json(
        client.cfg.model_smart,
        [{"role": "user", "content": CONFIRM_PROMPT.format(course=course, question=question, lines=numbered, candidates=cand_text)}],
        max_tokens=4000,
        reasoning_off=False,
    )
    calls += 1
    timings["confirm_s"] = confirm.latency_s
    flags = []
    for item in (confirm.data or {}).get("verdicts") or []:
        try:
            n = int(item.get("line"))
            conf = float(item.get("confidence") or 0)
        except (TypeError, ValueError):
            continue
        if item.get("is_mistake") and conf >= confirm_threshold and n in candidates:
            flags.append(Flag(n, str(item.get("reason") or ""), conf, str(item.get("hint") or "")))
    flags.sort(key=lambda f: f.line)
    return CheckReport(flags, algebra, sorted(candidates), timings, calls)


def link_continuations(lines: list[str | None]) -> list[str | None]:
    """Students write chains as `expr` / `= step` / `= step`. Give each `= step` its left-hand side
    (the previous line's last expression) so SymPy can compare consecutive steps."""
    out: list[str | None] = []
    for i, src in enumerate(lines):
        prev = out[i - 1] if i else None
        if src and src.lstrip().startswith("=") and prev:
            src = f"{prev.split('=')[-1].strip()} {src.strip()}"
        out.append(src)
    return out


def report_json(report: CheckReport) -> str:
    return json.dumps(
        {
            "flags": [f.__dict__ for f in report.flags],
            "screened": report.screened,
            "algebra": {n: c.verdict.value for n, c in report.algebra.items()},
            "timings": {k: round(v, 3) for k, v in report.timings.items()},
            "calls": report.calls,
        },
        indent=2,
    )
