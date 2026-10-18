"""The assignment the student is working on, and matching their iPad work to its questions.

Margin never watches the laptop. It gets the assignment in one of two ways:
  A. the student hands it over (drops a problem-set PDF; see assignment.py - read on the Mac), or
  B. the student asks ("hey Margin, look at my laptop"): ONE screenshot of the front-most window of a study app
     (browsers, PDF readers, docs - an allowlist, so Messages, Mail etc. are never captured) is read by
     Nemotron's vision model.

Each problem on the iPad is then matched to the question it answers, so checking, hints and conversation use
the real question ("A 5 kg block on a 37 degree incline...") instead of guessing from the handwriting.
"""

import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from margin.nebius import Nebius, image_part
from margin.session import norm

# Apps Margin may read. Anything else (Messages, Mail, Slack, Terminal...) is never captured.
STUDY_APPS = {
    "com.apple.Safari", "com.google.Chrome", "company.thebrowser.Browser", "org.mozilla.firefox",
    "com.microsoft.edgemac", "com.brave.Browser", "com.operasoftware.Opera",
    "com.apple.Preview", "com.adobe.Reader", "com.adobe.Acrobat.Pro", "com.readdle.PDFExpert-Mac",
    "net.sourceforge.skim-app.skim", "notion.id", "com.microsoft.Word", "com.apple.iWork.Pages",
    "com.apple.iBooksX", "md.obsidian",
}

READ_PROMPT = """This is a screenshot of a student's laptop window (e.g. a Canvas or Gradescope assignment, a PDF
problem set, lecture slides). Extract what they're being asked to do.

Return JSON:
{
  "course": "course code or name if visible (e.g. \\"PHYS 170\\"), else null",
  "title": "assignment / document title if visible, else null",
  "questions": [{"label": "the problem's number/letter as shown, e.g. \\"3b\\"", "text": "the full question, math in plain text"}]
}
Include every problem or question visible, in order. Use plain text math (x^2, sqrt(2), 37°). If there is no
assignment or question on screen, return an empty list. Do not answer the questions."""

MATCH_PROMPT = """A student is writing out a solution by hand. Which of these assignment questions are they working on?

Their handwritten lines:
{work}

Questions on their laptop screen:
{questions}

Return JSON: {{"label": "the matching question's label, or null if none fits", "confidence": 0-1}}"""


@dataclass
class LaptopQuestion:
    label: str
    text: str


@dataclass
class Laptop:
    app: str
    title: str | None
    course: str | None
    questions: list[LaptopQuestion]
    read_at: float = 0.0
    read_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Laptop":
        return cls(d.get("app", ""), d.get("title"), d.get("course"),
                   [LaptopQuestion(**q) for q in d.get("questions", [])], d.get("read_at", 0.0), d.get("read_s", 0.0))


# ------------------------------------------------------------------------------------------- capture


@dataclass
class Window:
    app: str
    bundle: str
    window_id: int
    title: str


def study_window() -> Window | None:
    """The front-most window that belongs to a study app (the one the student is looking at, even if Margin's
    terminal has focus). Windows of any other app are never returned."""
    import Quartz
    from AppKit import NSWorkspace

    apps = {a.processIdentifier(): a for a in NSWorkspace.sharedWorkspace().runningApplications()
            if a.bundleIdentifier() in STUDY_APPS}
    if not apps:
        return None
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for w in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:  # front to back
        bounds = w.get("kCGWindowBounds") or {}
        app = apps.get(w.get("kCGWindowOwnerPID"))
        if app and w.get("kCGWindowLayer", 0) == 0 and bounds.get("Width", 0) > 300 and bounds.get("Height", 0) > 200:
            return Window(app.localizedName(), app.bundleIdentifier(), int(w["kCGWindowNumber"]), w.get("kCGWindowName") or "")
    return None


def capture(window: Window) -> Image.Image | None:
    """Screenshot one window (needs Screen Recording permission for the terminal running Margin)."""
    path = Path(tempfile.gettempdir()) / "margin-laptop.png"
    result = subprocess.run(["screencapture", "-x", "-o", "-l", str(window.window_id), str(path)], capture_output=True)
    if result.returncode != 0 or not path.exists():
        return None
    try:
        img = Image.open(path)
        img.load()
        return img
    except OSError:
        return None


# ---------------------------------------------------------------------------------------------- read


def read_window(client: Nebius, img: Image.Image, app: str = "") -> Laptop:
    t = time.perf_counter()
    result = client.chat_json(
        client.cfg.model_vision,
        [{"role": "user", "content": [{"type": "text", "text": READ_PROMPT}, image_part(img, max_side=1600)]}],
        max_tokens=2000,
    )
    data = result.data or {}
    questions = [
        LaptopQuestion(str(q.get("label") or i + 1), str(q.get("text") or "").strip())
        for i, q in enumerate(data.get("questions") or [])
        if isinstance(q, dict) and q.get("text")
    ]
    return Laptop(app, data.get("title"), data.get("course"), questions, time.time(), time.perf_counter() - t)


_EQUATION = re.compile(r"[0-9a-z^+\-/().°]*=[0-9a-z^+\-/().°]+")


def _label_key(label) -> str:
    """"Q3", "#3", 3, "3)" -> "3"; "4(b)" -> "4b"."""
    return re.sub(r"^(q|question|problem|#)", "", re.sub(r"[^0-9a-z#]", "", str(label).lower()))


def _equations(text: str) -> set[str]:
    """The equations in a line, normalized: "Solve for x: 3x + 4 = 19." -> {"3x+4=19"}."""
    no_words = re.sub(r"\b[a-z]{2,}\b", " ", text.lower())  # drop words ("solve", "of"...); keep variables (x)
    return {e.strip(".") for e in _EQUATION.findall(norm(no_words)) if any(c.isdigit() for c in e)}


_LABEL = re.compile(r"^\s*(?:#|q|problem|question|ex(?:ercise)?)?\s*\(?(\d+\s*[a-z]?)\s*[.):]", re.I)


class QuestionMatcher:
    """Which laptop question is this iPad problem answering? Cheap rules first, Nemotron if needed; cached."""

    def __init__(self, client: Nebius | None):
        self.client = client
        self._cache: dict[tuple, LaptopQuestion | None] = {}

    def cached(self, work: list[str], laptop: Laptop | None) -> LaptopQuestion | None:
        return self._cache.get(self._key(work, laptop)) if laptop else None

    def match(self, work: list[str], laptop: Laptop | None) -> LaptopQuestion | None:
        if not laptop or not laptop.questions or not work:
            return None
        key = self._key(work, laptop)
        if key not in self._cache:
            self._cache[key] = self._match(work, laptop.questions)
        return self._cache[key]

    @staticmethod
    def _key(work: list[str], laptop: Laptop | None) -> tuple:
        return tuple(norm(w) for w in work[:3]), tuple(q.label for q in laptop.questions) if laptop else ()

    def _match(self, work: list[str], questions: list[LaptopQuestion]) -> LaptopQuestion | None:
        m = _LABEL.match(work[0])  # "3b) ..." / "#4." / "Problem 2:"
        if m:
            label = _label_key(m.group(1))
            for q in questions:
                if _label_key(q.label) == label:
                    return q
        # The problem copied from the screen: same equation ("Solve 5x - 2 = 13" vs "Solve for x: 5x - 2 = 13.")
        head_eqs = _equations(work[0])
        if head_eqs:
            hits = [q for q in questions if head_eqs & _equations(q.text)]
            if hits:
                return min(hits, key=lambda q: len(q.label))  # "4" over "4a" when both contain it
        if len(questions) == 1:
            return questions[0]
        if self.client is None:
            return None
        listing = "\n".join(f"[{q.label}] {q.text}" for q in questions)
        result = self.client.chat_json(
            self.client.cfg.model_fast,
            [{"role": "user", "content": MATCH_PROMPT.format(work="\n".join(work), questions=listing)}],
            max_tokens=100,
        )
        data = result.data or {}
        label = data.get("label")
        try:
            confident = float(data.get("confidence") or 0) >= 0.5
        except (TypeError, ValueError):
            confident = False
        if label is None or not confident:
            return None
        return next((q for q in questions if _label_key(q.label) == _label_key(label)), None)


def look_at_laptop(client: Nebius) -> tuple[Laptop | None, str]:
    """Option B: one screenshot of the study window, read by Nemotron. Returns (assignment, what happened)."""
    window = study_window()
    if window is None:
        return None, "I don't see a browser, PDF or document window open on your laptop."
    img = capture(window)
    if img is None:
        return None, ("I couldn't take a screenshot. Allow Screen Recording for your terminal in System Settings, "
                      "Privacy and Security, then restart me.")
    laptop = read_window(client, img, f"{window.app}: {window.title}" if window.title else window.app)
    if not laptop.questions:
        return laptop, f"I looked at your {window.app} window but didn't see any questions there."
    labels = ", ".join(q.label for q in laptop.questions[:6])
    return laptop, f"Got it, I can see {len(laptop.questions)} questions in {window.app}: {labels}."
