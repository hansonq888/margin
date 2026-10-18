"""Talking with Margin: answers spoken questions using the live session state.

The session state (problems, lines, circled lines, mistakes already found in the background) is turned
into a compact text page and given to Nemotron with the recent conversation. Replies stream, and are split
into sentences so the first one can be spoken while the rest is still being generated.
"""

import re
from collections.abc import Iterator

from margin.checker.algebra import Verdict, check_step
from margin.checker.cascade import link_continuations
from margin.coach import Coach
from margin.nebius import Nebius
from margin.session import Session

SYSTEM = """You are Margin, a warm, quick study companion. You can see the student's handwritten work on their
iPad (given below as text, read from their handwriting) and you have already checked it.

How to answer:
- You are SPEAKING out loud. Hard limit: 2 short sentences, under 35 words total. Plain conversational English.
  No LaTeX, no markdown, no lists, no symbols like ² or ^. Say math the way a person would ("x squared",
  "five x minus two", "thirteen plus two").
- Be Socratic: guide with a question or a pointed hint. NEVER state the answer or the corrected line unless they
  explicitly say "just tell me" or "what's the answer". Confusion or "I don't get it" is a reason for a smaller
  hint, not the answer.
- Their words come from speech recognition and can be garbled ("wait by" for "wait, why?"). If a question
  doesn't make sense, briefly ask them to say it again instead of guessing.
- Refer to their actual lines ("in your second line..."). If a mistake is listed under "Already found", and they
  ask for help or whether they're right, steer them to that line.
- If you can't see work related to their question, say so briefly and ask what they're working on.
- If the assignment is listed, use its wording; their work may skip restating the question.
- Lines marked [verified ...] were checked exactly by a computer algebra system: trust those over your own
  arithmetic. Never call a verified-correct line wrong. Do the arithmetic carefully if you must.
- If they're right, say so plainly and briefly."""

_GENERIC = re.compile(
    r"^(can you |could you |please )?(help|help me|help me out|i'?m stuck|am i (right|doing this right|on the right track)|"
    r"is this (right|correct)|check (my|this)( work)?|what'?s wrong|what did i do wrong)\b",
    re.I,
)


def is_generic_help(question: str) -> bool:
    q = re.sub(r"[^\w\s']", "", question).strip()
    q = re.sub(r"\b(with this( problem)?|here|please|real quick)\b", "", q, flags=re.I).strip()
    return bool(_GENERIC.match(q)) or q.lower() in {"", "help"}


_GOODBYE = re.compile(
    r"\b(bye|goodbye|see you|that'?s all|that'?s it|i'?m good|i'?m done|never ?mind|stop listening)\b|^(ok(ay)?,? )?thank(s| you)[.!]*$",
    re.I,
)


_LOOK = re.compile(
    r"\b(look|check|read|see|scan)\b.{0,25}\b(laptop|screen|computer|monitor|assignment|canvas|problem set|homework|pdf)\b",
    re.I,
)


def wants_look(question: str) -> bool:
    """"Hey Margin, look at my laptop" - the only time Margin takes a screenshot of the laptop."""
    return bool(_LOOK.search(question))


def is_goodbye(question: str) -> bool:
    """The student is wrapping up: answer, but don't keep listening for a follow-up."""
    return bool(_GOODBYE.search(question.strip()))


def page_context(session: Session, coach: Coach | None = None, matched=None) -> str:
    """The visible page as text for the model, with what's already been found. `matched(texts)` returns the
    assignment question a problem answers, if already known (no model calls here)."""
    focus = coach.focus[1] if coach and coach.focus else None
    out = []
    if session.laptop and session.laptop.questions:
        out.append(f"Assignment ({session.laptop.title or session.laptop.app}):")
        out += [f"  [{q.label}] {q.text}" for q in session.laptop.questions]
        out.append("")
    out.append("Their handwritten work on the iPad (lettered A, B, C... by position on the page; these letters are NOT")
    out.append("the assignment's question numbers - never tell the student they wrote a letter or number they didn't):")
    n = 0
    for block in sorted(session.blocks.values(), key=lambda b: (b.bbox[1], b.bbox[0])):
        for problem in block.problems():
            n += 1
            here = "  <- working on this now" if focus and problem[0].text == focus[0].text else ""
            out.append(f"Work {chr(64 + n) if n <= 26 else n}:{here}")
            q = matched([l.text for l in problem]) if matched else None
            if q:
                out.append(f"  (answering assignment question {q.label})")
            # Verified facts, so the spoken answer explains checked math instead of redoing it from memory.
            sym = link_continuations([l.sympy for l in problem])
            for i, line in enumerate(problem, 1):
                circled = "  (circled a ?)" if line is block.circled else ""
                verdict = check_step(sym[i - 2], sym[i - 1]).verdict if i > 1 else None
                checked = {Verdict.EQUIVALENT: "  [verified: follows correctly from the line above]",
                           Verdict.NOT_EQUIVALENT: "  [verified: does NOT follow from the line above]"}.get(verdict, "")
                out.append(f"  line {i}: {line.text}{circled}{checked}")
            for line in problem:
                f = session.flag_for(block, problem, line)
                if f:
                    out.append(f"  Already found: \"{line.text}\" - {f['reason']}")
    if n == 0:
        out.append("  (nothing written on the visible page yet)")
    return "\n".join(out)


def page_changes(before: str | None, after: str) -> list[str]:
    """What the student changed on their page since the last reply, as readable notes."""
    if before is None:
        return []
    line = re.compile(r"^\s+line \d+: (.*?)(\s+\(circled a \?\))?$")
    old = [m.group(1) for l in before.splitlines() if (m := line.match(l))]
    new = [m.group(1) for l in after.splitlines() if (m := line.match(l))]
    gone = [l for l in old if l not in new]
    added = [l for l in new if l not in old]
    notes = []
    for g, a in zip(gone, added):
        notes.append(f'changed "{g}" to "{a}"')
    notes += [f'erased "{g}"' for g in gone[len(added):]]
    notes += [f'wrote "{a}"' for a in added[len(gone):]]
    return notes


class Conversation:
    def __init__(self, client: Nebius, max_turns: int = 6, max_sentences: int = 2):
        self.client = client
        self.max_sentences = max_sentences
        self.history: list[dict] = []
        self.max_turns = max_turns
        self._last_page: str | None = None

    def messages(self, question: str, page: str, course: str = "unknown") -> list[dict]:
        """The page goes right next to the newest question (not at the top of the chat): it is the current
        truth and overrides anything said earlier, e.g. a line Margin talked about that has since been fixed."""
        changes = page_changes(self._last_page, page)
        changed = ("Since your last reply they " + "; ".join(changes) + ".\n\n") if changes else ""
        now = (f"[Their page RIGHT NOW - this replaces anything said earlier in the conversation]\n{page}\n\n"
               f"{changed}[They said] {question}")
        return [
            {"role": "system", "content": SYSTEM + f"\n\nCourse: {course}"},
            *self.history[-2 * self.max_turns :],
            {"role": "user", "content": now},
        ]

    def reply(self, question: str, page: str, course: str = "unknown") -> Iterator[str]:
        """Stream the answer, one sentence at a time (display text: LaTeX cleaned, math left as written)."""
        messages = self.messages(question, page, course)
        self._last_page = page
        answer = []
        for sentence in sentences(self.client.chat_stream(self.client.cfg.model_fast, messages, max_tokens=120)):
            answer.append(sentence)
            yield readable(sentence)
            if len(answer) >= self.max_sentences:
                break  # spoken answers must stay short, whatever the model wanted to add
        self.remember(question, " ".join(answer))

    def remember(self, question: str, answer: str) -> None:
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


_LATEX = [
    (re.compile(r"\\[()\[\]]|\$"), ""),  # \( \) \[ \] $
    (re.compile(r"\\(sin|cos|tan|log|ln|theta|alpha|beta|pi|sqrt|parallel|perp|cdot|times|pm)\b"), r" \1 "),
    (re.compile(r"\\frac\{([^}]*)\}\{([^}]*)\}"), r"\1 over \2"),
    (re.compile(r"\\[a-zA-Z]+"), " "),
    (re.compile(r"[{}]"), " "),
]
_LATEX_WORDS = {"cdot": "times", "times": "times", "pm": "plus or minus", "parallel": "parallel", "perp": "perpendicular",
                "sqrt": "the square root of"}

_SPOKEN = [
    (re.compile(r"([\w)])\s*\^\s*2\b|([\w)])²"), lambda m: (m.group(1) or m.group(2)) + " squared"),
    (re.compile(r"([\w)])\s*\^\s*3\b|([\w)])³"), lambda m: (m.group(1) or m.group(2)) + " cubed"),
    (re.compile(r"(\w)\s*\^\s*\(?(\w+)\)?"), lambda m: f"{m.group(1)} to the power {m.group(2)}"),
    (re.compile(r"(?<![\w)])[-–−](?=\s*\d)"), lambda m: "minus "),  # "-2" -> "minus 2"
    (re.compile(r"\s[-–−]\s(?=\d|\(|[a-zA-Z](?![a-zA-Z]))"), lambda m: " minus "),  # "x - 2", "16 - x"
    (re.compile(r"\s*[—–]\s*|\s+-\s+"), lambda m: ", "),  # a dash between words is a pause
    (re.compile(r"\s*=\s*"), lambda m: " equals "),
    (re.compile(r"\s*[×*]\s*"), lambda m: " times "),
    (re.compile(r"\s*÷\s*"), lambda m: " divided by "),
    (re.compile(r"√\s*\(?(\w+)\)?"), lambda m: f"the square root of {m.group(1)}"),
    (re.compile(r"±"), lambda m: "plus or minus "),
    (re.compile(r"\((minus \w+)\)"), lambda m: m.group(1)),  # "(minus 4) squared" -> "minus 4 squared"
    (re.compile(r"[*_`#]"), lambda m: ""),  # leftover markdown
]


def readable(text: str) -> str:
    """For the screen: strip LaTeX markup, keep math as written ("5x = 15")."""
    for pattern, repl in _LATEX:
        text = pattern.sub(repl, text)
    text = re.sub(r"\b(" + "|".join(_LATEX_WORDS) + r")\b", lambda m: _LATEX_WORDS[m.group(1)], text)
    text = re.sub(r"(\w)_(\w+)", r"\1 \2", text)  # F_parallel -> F parallel
    text = re.sub(r"[*`#]", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def speakable(text: str) -> str:
    """For text-to-speech: written math -> what a person would say. macOS reads "5x" as "five times",
    so a number followed by a variable gets a space, and the variable x is spelled "ex"."""
    text = readable(text)
    text = re.sub(r"(\d)([a-zA-Z])\b", r"\1 \2", text)  # 5x -> 5 x, 3y -> 3 y
    for pattern, repl in _SPOKEN:
        text = pattern.sub(repl, text)
    text = re.sub(r"(?<![\w'])x(?![\w'])", "ex", text)  # the variable x (not "x-ray", not inside words)
    return re.sub(r"\s{2,}", " ", text).strip()


def sentences(pieces: Iterator[str], min_len: int = 12) -> Iterator[str]:
    """Group streamed text into speakable sentences as early as possible."""
    buf = ""
    for piece in pieces:
        buf += piece
        while True:
            m = re.search(r"[.!?](\s+|$)", buf[min_len:])
            if not m or m.end() + min_len >= len(buf) and not m.group(1):
                break
            cut = min_len + m.end()
            yield buf[:cut].strip()
            buf = buf[cut:]
    if buf.strip():
        yield buf.strip()
