from concurrent.futures import Future

from margin.checker.cascade import CheckReport, Flag
from margin.coach import Coach
from margin.session import Session, norm
from margin.transcribe import Line, Transcription


def reading(*texts: str, circled: int | None = None, crossed: tuple[int, ...] = ()) -> Transcription:
    lines = [Line(t, None, None, "crossed_out" if i in crossed else "math", True, 0.9) for i, t in enumerate(texts)]
    return Transcription(lines, circled is not None, circled, raw=None)


def test_norm_ignores_spacing_and_marks():
    assert norm("x = 4") == norm("x=4") == norm("x=4 <-") == norm("X = 4 (?)")
    assert norm("x = 4") != norm("x = -4")


def test_rereading_a_block_replaces_it_and_keeps_its_id():
    s = Session()
    a = s.apply((100, 100, 300, 160), reading("2x + 3 = 1"))
    b = s.apply((100, 100, 320, 220), reading("2x + 3 = 11", "2x = 8"))
    assert a.id == b.id and b.version == 2
    assert list(s.blocks) == [a.id]
    assert [w.text for w in s.blocks[a.id].work()] == ["2x + 3 = 11", "2x = 8"]


def test_separate_blocks_coexist_and_scroll_together():
    s = Session()
    s.apply((100, 100, 300, 200), reading("x^2 = 9"))
    s.apply((600, 100, 800, 200), reading("2x = 8"))
    assert len(s.blocks) == 2
    s.scroll(0, -50)
    assert sorted(b.bbox[1] for b in s.blocks.values()) == [50, 50]


def test_empty_reading_clears_the_block():
    s = Session()
    s.apply((100, 100, 300, 200), reading("x = 1"))
    assert s.apply((100, 100, 300, 200), reading()) is None
    assert not s.blocks


def test_circled_line_maps_past_crossed_out_lines():
    s = Session()
    b = s.apply((0, 0, 10, 10), reading("a = 1", "junk", "b = 2", circled=2, crossed=(1,)))
    assert b.work_index(b.circled_line) == 2


class Immediate:
    """Stand-in check/help functions that record calls and return canned results."""

    def __init__(self, flags=None, hint="What do you get if you subtract 3?"):
        self.flags = flags or {}
        self.hint = hint
        self.checks, self.helps = [], []

    def check(self, question, work, course):
        self.checks.append([w.text for w in work])
        texts = [w.text for w in work]
        flags = [Flag(texts.index(t) + 1, "wrong", 0.9, f"hint for {t}") for t in self.flags if t in texts]
        return CheckReport(flags, {}, [])

    def help(self, question, work, line, course, trigger="circled"):
        self.helps.append(line)
        return {"hint": self.hint}


def make(fake, quiet=0.0, now=None, grace=0.0, proactive=True):
    now = now if now is not None else [0.0]
    s = Session()
    coach = Coach(
        s, check_fn=fake.check, help_fn=fake.help, quiet_seconds=quiet,
        grace_seconds=grace, last_line_grace_seconds=grace, clock=lambda: now[0], proactive=proactive,
    )
    return s, coach


def test_flags_a_mistake_once_and_skips_unchanged_blocks():
    fake = Immediate(flags={"x = 3"})
    s, coach = make(fake)
    block = s.apply((0, 0, 100, 100), reading("x^2 = 9", "x = 3"))
    coach.on_block(block)
    alerts = coach.drain()
    assert [(a.kind, a.line) for a in alerts] == [("mistake", 2)]
    # Same content re-read (e.g. pen hover): no new check, no repeat alert.
    coach.on_block(s.apply((0, 0, 100, 100), reading("x^2 = 9", "x = 3")))
    assert coach.drain() == [] and len(fake.checks) == 1


def test_stale_results_are_dropped_when_student_keeps_writing():
    fake = Immediate(flags={"x = 3"})
    s, coach = make(fake)
    block = s.apply((0, 0, 100, 100), reading("x^2 = 9", "x = 3"))
    gate = Future()
    coach.check_fn = lambda q, w, c: gate.result()
    coach.on_block(block)
    s.apply((0, 0, 100, 100), reading("x^2 = 9", "x = 3 or x = -3"))  # fixed before the check returned
    gate.set_result(CheckReport([Flag(2, "wrong", 0.9, "hint")], {}, []))
    assert coach.drain() == []


def test_circled_question_gets_help_even_during_quiet_period():
    now = [0.0]
    fake = Immediate(flags={"2x = ?"})
    s, coach = make(fake, quiet=60.0, now=now)
    coach.last_proactive = 0.0  # we just spoke
    block = s.apply((0, 0, 100, 100), reading("2x + 3 = 11", "2x = ?", circled=1))
    coach.on_block(block)
    alerts = coach.drain()
    assert [a.kind for a in alerts] == ["help"] and fake.helps == [2]
    # The help already covered that line, so no second (mistake) alert about it later.
    now[0] = 1000.0
    assert coach.poll() == []


def test_quiet_period_holds_proactive_alerts_and_drops_fixed_ones():
    now = [0.0]
    fake = Immediate(flags={"x = 3", "y = 5"})
    s, coach = make(fake, quiet=30.0, now=now)
    coach.on_block(s.apply((0, 0, 100, 100), reading("x^2 = 9", "x = 3")))
    assert len(coach.drain()) == 1
    coach.on_block(s.apply((500, 0, 600, 100), reading("y + 1 = 6", "y = 5")))
    now[0] = 10.0
    assert coach.drain() == []  # held: too soon after the last alert
    s.apply((500, 0, 600, 100), reading("y + 1 = 6", "y = 4"))  # student fixed it meanwhile
    now[0] = 100.0
    assert coach.poll() == []


def test_rereads_after_zoom_do_not_repeat_or_recheck():
    fake = Immediate(flags={"x = 3"})
    s, coach = make(fake)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve x^2 = 9", "x = 3")))
    assert [a.kind for a in coach.drain()] == ["mistake"]
    # Zoom: everything re-created with new ids, positions and slightly different text.
    s.clear()
    coach.on_block(s.apply((0, 0, 300, 300), reading("Solve x^2=9", "x=3"), replace=False))
    assert coach.drain() == [] and len(fake.checks) == 1


def test_circled_help_is_not_repeated_after_zoom():
    fake = Immediate()
    s, coach = make(fake)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve x^2 = 16", "x = 4", circled=1)))
    assert [a.kind for a in coach.drain()] == ["help"]
    s.clear()
    coach.on_block(s.apply((0, 0, 300, 300), reading("Solve x^2=16", "x=4 <-", circled=1), replace=False))
    assert coach.drain() == [] and fake.helps == [2]


def test_problems_in_one_block_are_checked_separately():
    fake = Immediate(flags={"5x = 11"})
    s, coach = make(fake)
    block = s.apply((0, 0, 100, 400), reading("Solve 3x + 4 = 19", "3x = 15", "x = 5", "Solve 5x - 2 = 13", "5x = 11"))
    assert [len(p) for p in block.problems()] == [3, 2]
    coach.on_block(block)
    alerts = coach.drain()
    assert [(a.line, a.line_text) for a in alerts] == [(2, "5x = 11")]
    assert sorted(len(c) for c in fake.checks) == [2, 3]
    # Writing a third problem below doesn't re-check the first two.
    block = s.apply((0, 0, 100, 500), reading("Solve 3x + 4 = 19", "3x = 15", "x = 5", "Solve 5x - 2 = 13", "5x = 11", "Solve x^2 = 1", "x = 1"))
    coach.on_block(block)
    coach.drain()
    assert len(fake.checks) == 3


def test_blocks_cut_off_at_the_side_are_not_judged():
    fake = Immediate(flags={"x = +4"})
    s, coach = make(fake)
    coach.on_block(s.apply((0, 0, 100, 100), reading("x^2 = 16", "x = +4", circled=1), clipped={"right"}))
    assert coach.drain() == [] and not fake.checks and not fake.helps


def test_only_the_problem_cut_off_at_the_top_is_skipped():
    fake = Immediate(flags={"5x = 11"})
    s, coach = make(fake)
    block = s.apply(
        (0, 0, 100, 400),
        reading("x = 5", "Solve 5x - 2 = 13", "5x = 11"),  # an older problem scrolled half off the top
        clipped={"top"},
    )
    coach.on_block(block)
    assert [a.line_text for a in coach.drain()] == ["5x = 11"]
    assert fake.checks == [["Solve 5x - 2 = 13", "5x = 11"]]


def test_page_rereads_do_not_erase_neighbours():
    s = Session()
    s.apply((100, 100, 400, 300), reading("Solve 5x - 2 = 13"), replace=False)
    s.apply((100, 290, 400, 500), reading("Solve x^2 - 16 = 0"), replace=False)  # padded boxes touch
    assert len(s.blocks) == 2


def test_half_written_last_line_is_not_called_a_mistake():
    now = [0.0]
    fake = Immediate(flags={"3x = 1"})
    s, coach = make(fake, now=now)
    coach.last_line_grace_seconds = 6.0
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 3x + 4 = 19", "3x = 1")))
    assert coach.drain() == []  # waiting: they may still be writing
    now[0] = 2.0
    s.apply((0, 0, 100, 100), reading("Solve 3x + 4 = 19", "3x = 15"))  # it was "15" all along
    now[0] = 10.0
    assert coach.poll() == []


def test_mistake_on_last_line_is_spoken_after_the_grace_period():
    now = [0.0]
    fake = Immediate(flags={"5x = 11"})
    s, coach = make(fake, now=now)
    coach.last_line_grace_seconds = 6.0
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 5x - 2 = 13", "5x = 11")))
    assert coach.drain() == []
    now[0] = 7.0
    assert [a.line_text for a in coach.poll()] == ["5x = 11"]


def test_ask_mode_stays_silent_until_asked_then_answers_instantly():
    fake = Immediate(flags={"5x = 11"})
    s, coach = make(fake, proactive=False)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 5x - 2 = 13", "5x = 11")))
    assert coach.drain() == []  # found it, kept quiet
    alerts = coach.ask()  # "hey Margin, can you help?"
    assert [(a.kind, a.line_text) for a in alerts] == [("help", "5x = 11")]
    assert not fake.helps  # answered from the background check, no extra model call


def test_ask_with_no_known_mistake_asks_the_tutor_about_the_latest_line():
    fake = Immediate(hint="What do you do next to isolate x?")
    s, coach = make(fake, proactive=False)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 5x - 2 = 13", "5x = 15")))
    coach.drain()
    assert coach.ask() == []
    alerts = coach.drain()
    assert [(a.kind, a.message) for a in alerts] == [("help", "What do you do next to isolate x?")]
    assert fake.helps == [2]


def test_ask_targets_the_problem_being_worked_on():
    fake = Immediate(flags={"3x = 1", "5x = 11"})
    s, coach = make(fake, proactive=False)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 3x + 4 = 19", "3x = 1")))
    coach.on_block(s.apply((0, 300, 100, 400), reading("Solve 5x - 2 = 13", "5x = 11")))
    coach.drain()
    assert [a.line_text for a in coach.ask()] == ["5x = 11"]


def test_ask_with_nothing_written():
    s, coach = make(Immediate(), proactive=False)
    assert "write something" in coach.ask()[0].message.lower()


def test_found_mistakes_survive_zoom_rereads():
    from margin.overlay import page_snapshot

    fake = Immediate(flags={"5x = 11"})
    s, coach = make(fake)
    coach.on_block(s.apply((0, 0, 100, 100), reading("Solve 5x - 2 = 13", "5x = 11")))
    coach.drain()
    s.clear()  # zoom: the page is re-read into new blocks, with slightly different text
    s.apply((0, 0, 300, 300), reading("Solve 5x-2=13", "5x=11"), replace=False)
    lines = page_snapshot(s)[0]["problems"][0]["lines"]
    assert lines[1]["flag"] and lines[1]["flag"]["reason"] == "wrong"
    again = Session.from_dict(s.to_dict())
    assert again.flags == s.flags
