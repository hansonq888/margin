"""Handwriting crop -> structured lines, via Nemotron Nano Omni."""

from dataclasses import dataclass

from PIL import Image

from margin.nebius import Nebius, Result, image_part

PROMPT = """You are reading a student's handwritten notes from an iPad. Transcribe ONLY what is written in the image.

Return a JSON object:
{
  "lines": [
    {
      "text": "plain-text reading of the line",
      "latex": "LaTeX for math lines, else null",
      "sympy": "SymPy-parsable form if the line is pure algebra/arithmetic, else null",
      "kind": "math" | "text" | "diagram" | "crossed_out",
      "complete": true if the line looks finished, false if it trails off mid-expression,
      "new_problem": true if this line starts a new problem (e.g. "Solve ...", a problem number, a fresh unrelated equation), else false,
      "confidence": number from 0 to 1 for how sure you are of the reading
    }
  ],
  "circled_question_mark": {"present": true/false, "near_line": index into lines or null}
}

Rules for "sympy":
- Use = for equations, ** for powers, explicit * for multiplication, sqrt(), sin(), exp(), log(), pi.
- Angles written in degrees become radians: sin 30° -> sin(30*pi/180).
- Drop units. Use plain variable names (theta, n_H2O, F_g).
- Use null for derivatives/integrals written in Leibniz or prime notation, prose, or anything ambiguous.

Do not put a circled question mark or arrows into any line's text; report the circled "?" only in circled_question_mark.
Do not solve, correct, or comment on the work. If a word or symbol is illegible, give your best guess and lower the confidence."""


@dataclass
class Line:
    text: str
    latex: str | None
    sympy: str | None
    kind: str
    complete: bool
    confidence: float
    new_problem: bool = False


@dataclass
class Transcription:
    lines: list[Line]
    circled_question: bool
    circled_near_line: int | None
    raw: Result


def transcribe(client: Nebius, crop: Image.Image, model: str | None = None) -> Transcription:
    result = client.chat_json(
        model or client.cfg.model_vision,
        [{"role": "user", "content": [{"type": "text", "text": PROMPT}, image_part(crop)]}],
        max_tokens=1200,
    )
    data = result.data or {}
    lines = []
    for item in data.get("lines") or []:
        if not isinstance(item, dict):
            continue
        lines.append(
            Line(
                text=str(item.get("text") or ""),
                latex=item.get("latex"),
                sympy=item.get("sympy"),
                kind=str(item.get("kind") or "text"),
                complete=bool(item.get("complete", True)),
                confidence=_as_float(item.get("confidence"), 0.5),
                new_problem=bool(item.get("new_problem", False)),
            )
        )
    circled = data.get("circled_question_mark") or {}
    return Transcription(
        lines=lines,
        circled_question=bool(circled.get("present", False)),
        circled_near_line=circled.get("near_line"),
        raw=result,
    )


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
