"""Loading a problem set the student hands to Margin (option A: nothing on screen is watched).

A normal PDF is read entirely on the Mac: its text is extracted and split into numbered questions locally.
Only a scanned PDF (pages are images, no text) needs Nemotron's vision model, and Margin says so before
sending those pages.
"""

import re
from pathlib import Path

from margin.laptop import Laptop, LaptopQuestion

_COURSE = re.compile(r"\b([A-Z]{2,5})\s?-?(\d{2,4}[A-Z]?)\b")
# "1." "2)" "3:" "Q4" "Problem 5" "Exercise 6" at the start of a line
_QUESTION = re.compile(r"^\s*(?:(?:problem|question|exercise|ex\.?|q)\s*)?(\d{1,3})\s*[.):]\s+(.*)$", re.I)
_PART = re.compile(r"^\s*\(?([a-h])[.)]\s+(.*)$")


def split_questions(text: str) -> list[LaptopQuestion]:
    """Split problem-set text into questions. Parts "(a)", "(b)" become their own questions ("4a", "4b")
    that carry the parent's text, so "check part b" still has the full context."""
    questions: list[LaptopQuestion] = []
    current: LaptopQuestion | None = None
    parent_text = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        q = _QUESTION.match(line)
        p = _PART.match(line)
        if q:
            current = LaptopQuestion(q.group(1), q.group(2).strip())
            parent_text = current.text
            questions.append(current)
        elif p and current is not None:
            number = re.match(r"\d+", current.label).group()
            current = LaptopQuestion(f"{number}{p.group(1)}", f"{parent_text} ({p.group(1)}) {p.group(2).strip()}")
            questions.append(current)
        elif current is not None:
            current.text = f"{current.text} {line.strip()}"
            if current.label.isdigit():
                parent_text = current.text
    return questions


def guess_course(text: str) -> str | None:
    for m in _COURSE.finditer(text[:600]):
        if m.group(1) not in {"PDF", "PM", "AM", "PST", "PDT", "EST", "EDT"}:
            return f"{m.group(1)} {m.group(2)}"
    return None


def load_pdf(path: str | Path, client=None, allow_cloud: bool = True) -> Laptop:
    """Read a problem set. Text PDFs never leave the Mac; scanned pages are read by Nemotron VL."""
    from pypdf import PdfReader

    path = Path(path).expanduser()
    reader = PdfReader(str(path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    title = next((l.strip() for l in text.splitlines() if l.strip()), path.stem)
    questions = split_questions(text)
    if questions:
        return Laptop(f"PDF (read on this Mac): {path.name}", title, guess_course(text), questions)

    if not (allow_cloud and client is not None):
        return Laptop(f"PDF: {path.name}", path.stem, None, [])
    # Scanned: no text layer. Render pages and read them with the vision model.
    import pypdfium2 as pdfium

    from margin.laptop import read_window

    print(f"[assignment] {path.name} is scanned (no text); sending its pages to Nemotron to read them.")
    doc = pdfium.PdfDocument(str(path))
    found: list[LaptopQuestion] = []
    course = None
    for i in range(min(len(doc), 6)):
        img = doc[i].render(scale=1.5).to_pil()
        page = read_window(client, img, path.name)
        found += page.questions
        course = course or page.course
    return Laptop(f"PDF (scanned, read by Nemotron): {path.name}", path.stem, course, found)
