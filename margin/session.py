"""The live session state: one model of what's on the visible page, built from many partial readings.

The ink tracker emits a region per block of writing each time it changes; each region is read in full.
So a new reading *replaces* whatever blocks it overlaps instead of being appended. A block can hold several
problems written close together; `Block.problems()` splits them.

Re-reads of the same ink vary a little ("x = 4" vs "x=4"), and zooming re-creates every block, so anything
that must be remembered across readings (what was checked, what was said) is keyed by normalized *content*
via `norm()`, never by block id or position.
"""

import json
import re
from dataclasses import asdict, dataclass, field

from margin.transcribe import Line, Transcription

Box = tuple[int, int, int, int]

_NOISE = re.compile(r"\s+|\(\?\)|\?|<-+|->|←|→|\*|·|\.$")
_PROBLEM_START = re.compile(r"^\s*(solve|find|show|prove|compute|evaluate|simplify|q\s*\d|\(?\d+[a-z]?[.)])", re.I)


def norm(text: str) -> str:
    """Canonical form of a line for comparing re-reads: case, spacing and stray marks don't matter."""
    return _NOISE.sub("", text.lower()).replace("−", "-").replace("×", "x")


@dataclass
class Block:
    id: int
    bbox: Box
    lines: list[Line]
    version: int = 1
    circled_line: int | None = None  # index into `lines`
    clipped: frozenset = frozenset()  # screen edges the writing runs off; problems there may be incomplete
    flags: list[dict] = field(default_factory=list)

    def work(self) -> list[Line]:
        """Lines that count as the student's work (crossed-out and empty lines dropped)."""
        return [line for line in self.lines if line.kind != "crossed_out" and line.text.strip()]

    def work_index(self, line_index: int | None) -> int | None:
        """Map an index into `lines` to a 1-based position in `work()`."""
        if line_index is None or not 0 <= line_index < len(self.lines):
            return None
        target = self.lines[line_index]
        for n, line in enumerate(self.work(), 1):
            if line is target:
                return n
        return None

    @property
    def circled(self) -> Line | None:
        if self.circled_line is None or not 0 <= self.circled_line < len(self.lines):
            return None
        return self.lines[self.circled_line]

    def problems(self) -> list[list[Line]]:
        """Split the work into problems at lines that start a new one."""
        out: list[list[Line]] = []
        for line in self.work():
            if not out or (line.new_problem or _PROBLEM_START.match(line.text)) and out[-1]:
                out.append([])
            out[-1].append(line)
        return out

    @property
    def partial(self) -> bool:
        return bool(self.clipped)

    def judgeable_problems(self) -> list[list[Line]]:
        """Problems that are fully on screen. Cut off at the top: the first problem may be missing its start;
        at the bottom: the last may be missing its end; at a side: any line may be cut, so judge nothing."""
        problems = self.problems()
        if {"left", "right"} & self.clipped:
            return []
        if "top" in self.clipped:
            problems = problems[1:]
        if "bottom" in self.clipped:
            problems = problems[:-1]
        return problems

    def signature(self) -> tuple:
        return tuple(norm(line.text) for line in self.work())


def problem_signature(problem: list[Line]) -> tuple:
    return tuple(norm(line.text) for line in problem)


class Session:
    def __init__(self, course: str = "unknown"):
        self.course = course
        self.blocks: dict[int, Block] = {}
        self._next_id = 1
        self.talk: list[dict] = []  # spoken exchanges with Margin, most recent last
        self.laptop = None  # margin.laptop.Laptop: the assignment the student handed over / showed
        # Mistakes found, keyed by content (problem's first line, the line) so they survive re-reads and zooms.
        self.flags: dict[tuple[str, str], dict] = {}

    def flag_for(self, block: "Block", problem: list[Line], line: Line) -> dict | None:
        found = self.flags.get((norm(problem[0].text), norm(line.text)))
        if found:
            return found
        return next((f for f in block.flags if norm(f["line"]) == norm(line.text)), None)

    def clear(self) -> None:
        self.blocks.clear()

    def scroll(self, dx: int, dy: int) -> None:
        for b in self.blocks.values():
            x0, y0, x1, y1 = b.bbox
            b.bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

    def apply(self, bbox: Box, reading: Transcription, replace: bool = True, clipped: frozenset = frozenset()) -> Block | None:
        """Store a reading of one block. With replace=True it supersedes every block it overlaps (an edit);
        page re-reads after a zoom pass replace=False since their padded boxes can touch neighbours."""
        replaced = [b for b in self.blocks.values() if overlaps(b.bbox, bbox)] if replace else []
        for b in replaced:
            del self.blocks[b.id]
        if not reading.lines:
            return None
        if replaced:
            block_id = min(b.id for b in replaced)
            version = max(b.version for b in replaced) + 1
        else:
            block_id, version = self._next_id, 1
            self._next_id += 1
        circled = reading.circled_near_line if reading.circled_question else None
        if circled is not None and not (isinstance(circled, int) and 0 <= circled < len(reading.lines)):
            circled = len(reading.lines) - 1  # present but unplaced: assume the latest line
        block = Block(block_id, bbox, list(reading.lines), version, circled, frozenset(clipped))
        self.blocks[block_id] = block
        return block

    def problem_signatures(self) -> set[tuple]:
        return {problem_signature(p) for b in self.blocks.values() for p in b.problems()}

    def find_line(self, head: str, line: str) -> Block | None:
        """The block currently showing `line` in the problem that starts with `head` (both normalized)."""
        for b in self.blocks.values():
            for p in b.problems():
                if norm(p[0].text) == head and any(norm(l.text) == line for l in p):
                    return b
        return None

    def to_dict(self) -> dict:
        return {
            "course": self.course,
            "blocks": [
                {
                    "id": b.id,
                    "version": b.version,
                    "bbox": list(b.bbox),
                    "clipped": sorted(b.clipped),
                    "problems": [[l.text for l in p] for p in b.problems()],
                    "lines": [asdict(line) for line in b.lines],
                    "circled_line": b.circled_line,
                    "flags": b.flags,
                }
                for b in sorted(self.blocks.values(), key=lambda b: (b.bbox[1], b.bbox[0]))
            ],
            "found": [{"problem": k[0], **v} for k, v in self.flags.items()],
            "talk": self.talk[-10:],
            "assignment": self.laptop.to_dict() if self.laptop else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Rebuild a session from `to_dict()` output (saved sessions, demos, tests)."""
        s = cls(course=data.get("course", "unknown"))
        for b in data.get("blocks", []):
            lines = [Line(**line) for line in b["lines"]]
            block = Block(b["id"], tuple(b["bbox"]), lines, b.get("version", 1), b.get("circled_line"),
                          frozenset(b.get("clipped", [])), list(b.get("flags", [])))
            s.blocks[block.id] = block
            s._next_id = max(s._next_id, block.id + 1)
        s.talk = list(data.get("talk", []))
        for f in data.get("found", []):
            f = dict(f)
            s.flags[(f.pop("problem"), norm(f["line"]))] = f
        if data.get("assignment"):
            from margin.laptop import Laptop

            s.laptop = Laptop.from_dict(data["assignment"])
        return s

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def overlaps(a: Box, b: Box, slack: int = 8) -> bool:
    return a[0] < b[2] + slack and b[0] < a[2] + slack and a[1] < b[3] + slack and b[1] < a[3] + slack
