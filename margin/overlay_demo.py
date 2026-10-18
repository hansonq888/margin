"""Preview the overlay without an iPad or a mic: plays a scripted exchange over the demo page, on a loop.

    uv run python -m margin.overlay_demo            # then open http://127.0.0.1:8765/
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from margin.assignment import load_pdf
from margin.checker.cascade import Flag  # noqa: F401  (documents the flag shape)
from margin.ink import crop_to_ink
from margin.laptop import QuestionMatcher
from margin.overlay import Overlay, page_snapshot
from margin.session import Session

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "data"
NOTES = ROOT / "experiments" / "notes"

SCRIPT = [
    ("status", "listening", "say “Hey Margin…”"),
    ("pause", 1.5),
    ("you", "Hey Margin, is my second problem right?"),
    ("pause", 1.0),
    ("margin", "Close, but look at your second line."),
    ("highlight", "5x = 11", "mistake"),
    ("margin", "What do you get when you add two to thirteen?"),
    ("done",),
    ("status", "listening", "follow-up"),
    ("pause", 3.0),
    ("you", "Oh, fifteen. And the last one?"),
    ("pause", 1.0),
    ("margin", "You found x equals 4, nice."),
    ("highlight", "x = 4", "help"),
    ("margin", "What other number squared gives sixteen?"),
    ("done",),
    ("status", "watching", ""),
    ("pause", 5.0),
]


def build(overlay: Overlay) -> Session:
    session = Session.from_dict(json.loads((DATA / "demo_session.json").read_text()))
    session.laptop = load_pdf(DATA / "sample_problem_set_text.pdf")
    session.course = session.laptop.course or session.course
    for block in session.blocks.values():  # real readings come with LaTeX; give the demo page some too
        for line in block.lines:
            if not line.latex:
                word, _, rest = line.text.partition(" ")
                line.latex = rf"\text{{{word} }}{rest}" if word.isalpha() and len(word) > 1 else line.text
    matcher = QuestionMatcher(None)
    by_first = {b.lines[0].text: b for b in session.blocks.values()}
    for first, note in [("Solve 3x + 4 = 19", "Margin Test-1.jpg")]:  # the only demo problem with a matching photo
        if first in by_first and (NOTES / note).exists():
            overlay.set_crop(by_first[first].id, crop_to_ink(Image.open(NOTES / note)))
    last = max(session.blocks.values(), key=lambda b: b.bbox[1])
    focus = (last.id, last.problems()[-1])

    class _Coach:  # page_snapshot only needs .focus
        pass

    coach = _Coach()
    second = next(b for b in session.blocks.values() if b.lines[0].text.startswith("Solve 5x"))
    coach.focus = (second.id, second.problems()[0])
    overlay.update(
        course=session.course,
        assignment={"title": session.laptop.title, "count": len(session.laptop.questions)},
        page=page_snapshot(session, coach, matched=lambda t: matcher.match(t, session.laptop)),
        privacy={"voice": "on this Mac", "ipad": True},
    )
    overlay._demo = (session, coach, matcher, focus)
    return session


def play(overlay: Overlay, speed: float = 1.0) -> None:
    session, coach, matcher, focus = overlay._demo
    while True:
        overlay.update(turns=[], highlight=None)
        for step in SCRIPT:
            kind = step[0]
            if kind == "pause":
                time.sleep(step[1] / speed)
            elif kind == "status":
                overlay.set_status(step[1], step[2])
            elif kind == "you":
                overlay.heard(step[1])
            elif kind == "margin":
                overlay.set_status("speaking")
                overlay.margin_says(step[1])
                time.sleep(1.6 / speed)
            elif kind == "highlight":
                block = next(b for b in session.blocks.values() if any(l.text == step[1] for l in b.lines))
                if step[1] == "x = 4":
                    coach.focus = focus
                    overlay.update(page=page_snapshot(session, coach, matched=lambda t: matcher.match(t, session.laptop)))
                overlay.highlight(block.id, step[1], step[2])
            elif kind == "done":
                overlay.margin_says("", done=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    overlay = Overlay(port=args.port)
    build(overlay)
    print(f"Overlay demo at {overlay.start()}", flush=True)
    play(overlay, args.speed)


if __name__ == "__main__":
    main()
