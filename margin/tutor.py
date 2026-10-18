"""Socratic help for a specific line the student asked about (e.g. by circling a "?")."""

from margin.checker.cascade import WorkLine
from margin.nebius import Nebius

TRIGGERS = {
    "circled": "circled a question mark next to a line of their handwritten work, meaning they are stuck or unsure there",
    "asked": "just asked out loud \"can you help me?\" while working on this problem",
}

HELP_PROMPT = """A student {trigger}. Course: {course}.
The question they are answering: {question}

Work (numbered lines):
{lines}

The line in question is line {line}: "{text}"

Give ONE short Socratic hint (one sentence, spoken aloud) that helps them take the next step themselves.
Do not give the final answer. If line {line} contains a mistake, steer them toward noticing it.
Return JSON: {{"hint": "...", "has_mistake": true|false}}"""


def help_with_line(
    client: Nebius, question: str, work: list[WorkLine], line: int, course: str = "unknown", trigger: str = "circled"
) -> dict:
    numbered = "\n".join(f"{i}. {w.text}" for i, w in enumerate(work, 1))
    text = work[line - 1].text if 0 < line <= len(work) else ""
    result = client.chat_json(
        client.cfg.model_smart,
        [{"role": "user", "content": HELP_PROMPT.format(trigger=TRIGGERS.get(trigger, TRIGGERS["circled"]), course=course, question=question,
                                          lines=numbered, line=line, text=text)}],
        max_tokens=3000,
        reasoning_off=False,
    )
    data = result.data or {}
    return {
        "hint": str(data.get("hint") or "").strip(),
        "has_mistake": bool(data.get("has_mistake", False)),
        "latency_s": result.latency_s,
        "error": result.error,
    }
