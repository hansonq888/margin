"""The coach watches the session and decides when to speak.

- Each problem is re-checked in the background (SymPy -> Nemotron cascade) whenever its work changes.
- A circled "?" asks for help on that line; that bypasses the interruption budget (the student asked).
- Memory is keyed by content (problem's first line + the line, normalized), so re-reads, zooms and block
  re-creation never make it repeat itself or re-check work it has already seen.
- Proactive mistake alerts wait a grace period (longer for the line being written last, which may just be
  unfinished: "3x = 1" on the way to "3x = 15"), respect a quiet period, and are dropped if the line changes.
- Problems cut off by the screen edge are never judged: the reading may be missing pieces.

Two modes:
- proactive=True ("coach"): confirmed mistakes are spoken unprompted, within the budget above.
- proactive=False ("ask", default): everything is still checked in the background, but Margin only speaks
  when asked (circled "?", or `ask()` from a hotkey / "hey Margin"). Because the checking already happened,
  the answer is instant and points at the exact line.
"""

import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable

from margin.checker.cascade import CheckReport, WorkLine, check
from margin.session import Block, Session, norm, problem_signature


@dataclass
class Alert:
    block_id: int
    line: int  # 1-based position within the problem
    line_text: str
    kind: str  # "mistake" | "help"
    message: str  # what to say
    reason: str = ""
    head: str = ""  # normalized first line of the problem
    ready_at: float = 0.0  # don't speak before this time (grace period)


@dataclass
class _Job:
    kind: str
    future: Future
    block_id: int
    head: str
    texts: list[str]
    signature: tuple
    line: int | None = None


CheckFn = Callable[[str, list[WorkLine], str], CheckReport]
HelpFn = Callable[[str, list[WorkLine], int, str, str], dict]  # question, work, line, course, trigger


class Coach:
    def __init__(
        self,
        session: Session,
        client=None,
        check_fn: CheckFn | None = None,
        help_fn: HelpFn | None = None,
        quiet_seconds: float = 20.0,
        grace_seconds: float = 2.0,
        last_line_grace_seconds: float = 6.0,
        clock: Callable[[], float] = time.monotonic,
        workers: int = 3,
        proactive: bool = False,
        question_for: Callable[[list[str]], str | None] | None = None,
    ):
        if check_fn is None or help_fn is None:
            from margin.tutor import help_with_line

            check_fn = check_fn or (lambda q, work, course: check(client, q, work, course=course))
            help_fn = help_fn or (lambda q, work, line, course, trigger: help_with_line(client, q, work, line, course, trigger))
        self.session = session
        self.check_fn = check_fn
        self.help_fn = help_fn
        self.quiet_seconds = quiet_seconds
        self.grace_seconds = grace_seconds
        self.last_line_grace_seconds = last_line_grace_seconds
        self.clock = clock
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.jobs: list[_Job] = []
        self.held: list[Alert] = []  # proactive alerts waiting out the quiet period
        self.checked: set[tuple] = set()  # problem signatures already sent for checking
        self.spoken: set[tuple[str, str]] = set()  # (problem head, line) already addressed
        self.last_proactive = float("-inf")
        self.proactive = proactive
        self.question_for = question_for  # problem lines -> the assignment question they answer (or None)
        self.focus: tuple[int, list] | None = None  # (block id, problem lines) the student touched most recently

    def on_block(self, block: Block) -> None:
        circled = block.circled
        for problem in block.judgeable_problems():
            head = norm(problem[0].text)
            texts = [line.text for line in problem]
            lines = [WorkLine(line.text, line.sympy) for line in problem]
            sig = problem_signature(problem)

            if circled is not None and any(line is circled for line in problem):
                n = next(i for i, line in enumerate(problem, 1) if line is circled)
                key = (head, norm(circled.text))
                if key not in self.spoken:
                    self.spoken.add(key)
                    fut = self.pool.submit(self._help, texts, lines, n, "circled")
                    self.jobs.append(_Job("help", fut, block.id, head, texts, sig, n))

            if sig not in self.checked or (circled is not None and any(line is circled for line in problem)):
                self.focus = (block.id, problem)
            if sum(1 for line in problem if line.kind == "math") >= 2 and sig not in self.checked:
                self.checked.add(sig)
                fut = self.pool.submit(self._check, texts, lines)
                self.jobs.append(_Job("check", fut, block.id, head, texts, sig))

    def poll(self) -> list[Alert]:
        """Collect finished background work. Call this often (every frame is fine)."""
        out: list[Alert] = []
        for job in [j for j in self.jobs if j.future.done()]:
            self.jobs.remove(job)
            try:
                result = job.future.result()
            except Exception as e:  # a failed model call must never take the session down
                print(f"[coach] {job.kind} failed: {e}", file=sys.stderr)
                continue
            if job.kind == "help":
                text = job.texts[job.line - 1]
                block = self.session.find_line(job.head, norm(text))
                if block and result.get("hint"):
                    out.append(Alert(block.id, job.line, text, "help", result["hint"], head=job.head))
                continue
            if job.signature not in self.session.problem_signatures():
                continue  # stale: the student kept writing; a newer check is on its way
            for f in result.flags:
                if not 0 < f.line <= len(job.texts):
                    continue
                text = job.texts[f.line - 1]
                self.session.flags[(job.head, norm(text))] = {"line": text, "reason": f.reason, "hint": f.hint}
                key = (job.head, norm(text))
                if key in self.spoken:
                    continue
                self.spoken.add(key)
                grace = self.last_line_grace_seconds if f.line == len(job.texts) else self.grace_seconds
                self.held.append(
                    Alert(job.block_id, f.line, text, "mistake", f.hint or f.reason, f.reason, job.head, self.clock() + grace)
                )
        out += self._release_held()
        return out

    def ask(self) -> list[Alert]:
        """The student asked for help ("hey Margin, can you help?" / hotkey). Answer about what they were
        just working on: a mistake we already found there, or else a hint for their latest line."""
        if self.focus is None:
            return [Alert(0, 0, "", "help", "I don't see any work yet. Write something and ask me again.")]
        found = self.take_focus_mistake()
        if found:
            return [found]
        block_id, problem = self.focus
        head = norm(problem[0].text)
        n = len(problem)
        self.spoken.add((head, norm(problem[-1].text)))
        lines = [WorkLine(line.text, line.sympy) for line in problem]
        fut = self.pool.submit(self._help, [line.text for line in problem], lines, n, "asked")
        self.jobs.append(_Job("help", fut, block_id, head, [line.text for line in problem], problem_signature(problem), n))
        return []

    def question(self, texts: list[str]) -> str:
        """The question this work answers: from the assignment if we have it, else the work's first line.
        (Runs in a worker thread; matching may call Nemotron the first time.)"""
        if self.question_for:
            try:
                found = self.question_for(texts)
            except Exception as e:
                print(f"[coach] question matching failed: {e}", file=sys.stderr)
                found = None
            if found:
                return found
        return texts[0]

    def _check(self, texts: list[str], lines: list[WorkLine]) -> CheckReport:
        return self.check_fn(self.question(texts), lines, self.session.course)

    def _help(self, texts: list[str], lines: list[WorkLine], n: int, trigger: str) -> dict:
        return self.help_fn(self.question(texts), lines, n, self.session.course, trigger)

    def recheck(self) -> None:
        """The assignment changed: check every visible problem again against its real question.
        (What was already said stays said.)"""
        self.checked.clear()
        for block in list(self.session.blocks.values()):
            self.on_block(block)

    def take_focus_mistake(self) -> Alert | None:
        """A mistake already found (and not yet mentioned) in the problem the student is working on."""
        if self.focus is None:
            return None
        head = norm(self.focus[1][0].text)
        found = sorted((a for a in self.held if a.head == head), key=lambda a: a.line)
        if not found:
            return None
        alert = found[0]
        self.held.remove(alert)
        alert.kind = "help"
        return alert

    def _release_held(self) -> list[Alert]:
        released = []
        for alert in list(self.held):
            block = self.session.find_line(alert.head, norm(alert.line_text))
            if block is None:
                self.held.remove(alert)  # they fixed, finished or erased it
                self.spoken.discard((alert.head, norm(alert.line_text)))  # if it comes back, it's fair game again
                continue
            if not self.proactive:
                continue  # ask mode: keep it ready for when they ask
            if self.clock() >= alert.ready_at and self.clock() - self.last_proactive >= self.quiet_seconds:
                self.held.remove(alert)
                self.last_proactive = self.clock()
                alert.block_id = block.id
                released.append(alert)
        return released

    def drain(self, timeout: float = 60.0) -> list[Alert]:
        """Wait for outstanding work (used at the end of a replay)."""
        wait([j.future for j in self.jobs], timeout=timeout)
        return self.poll()

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
