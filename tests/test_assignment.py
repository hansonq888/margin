from margin.assignment import guess_course, load_pdf, split_questions
from margin.coach import Coach
from margin.converse import page_context
from margin.laptop import Laptop, LaptopQuestion, QuestionMatcher
from margin.session import Session
from margin.transcribe import Line, Transcription

PS = """MATH 112 - Problem Set 4
Due Friday, Oct 24. Show all work.

1. Solve for x: 3x + 4 = 19.
2) Solve for x: 5x - 2 = 13.
Problem 3: A 5 kg block rests on a frictionless ramp inclined at 37 degrees.
   Find the component of its weight along the ramp (g = 9.8 m/s^2).
4. Find all real solutions of x^2 - 16 = 0.
   (a) Solve by factoring.
   (b) Check both answers.
"""


def test_split_questions():
    qs = split_questions(PS)
    assert [q.label for q in qs] == ["1", "2", "3", "4", "4a", "4b"]
    assert qs[2].text.endswith("(g = 9.8 m/s^2).")  # wrapped line joined
    assert "x^2 - 16 = 0" in qs[5].text and "(b) Check both answers." in qs[5].text  # part keeps parent context


def test_guess_course():
    assert guess_course(PS) == "MATH 112"
    assert guess_course("CHEM-161 Lab 2") == "CHEM 161"
    assert guess_course("Due 5 PM 2026") is None


def test_load_text_pdf_stays_local():
    lap = load_pdf("experiments/data/sample_problem_set_text.pdf", client=None)
    assert lap.course == "MATH 112" and "read on this Mac" in lap.app
    assert [q.label for q in lap.questions] == ["1", "2", "3", "4", "4a", "4b"]


LAP = Laptop("PDF", "PS4", "MATH 112", split_questions(PS))


def test_matcher_rules_need_no_model():
    m = QuestionMatcher(client=None)
    assert m.match(["2) 5x - 2 = 13", "5x = 15"], LAP).label == "2"  # numbered on the iPad
    assert m.match(["Solve for x: 5x - 2 = 13", "5x = 15"], LAP).label == "2"  # question copied over
    assert m.match(["Solve 3x + 4 = 19", "3x = 15"], LAP).label == "1"  # copied loosely
    assert m.match(["Solve x^2 - 16 = 0", "x^2 = 16"], LAP).label == "4"  # parent, not 4a/4b
    assert m.match(["F = mg sin θ", "F = 5(9.8) cos 37°"], LAP) is None  # needs the model (no client here)
    assert m.match(["anything"], Laptop("x", None, None, [LaptopQuestion("1", "Only one")])).label == "1"
    assert m.cached(["2) 5x - 2 = 13", "5x = 15"], LAP).label == "2"


def test_coach_checks_against_the_assignment_question():
    seen = []
    s = Session()
    s.laptop = LAP
    m = QuestionMatcher(client=None)
    coach = Coach(
        s,
        check_fn=lambda q, work, course: seen.append(q) or __import__("margin.checker.cascade", fromlist=["x"]).CheckReport([], {}, []),
        help_fn=lambda *a: {"hint": ""},
        question_for=lambda texts: (q.text if (q := m.match(texts, s.laptop)) else None),
    )
    lines = [Line(t, None, None, "math", True, 0.9) for t in ["Solve for x: 5x - 2 = 13", "5x = 15"]]
    coach.on_block(s.apply((0, 0, 10, 10), Transcription(lines, False, None, raw=None)))
    coach.drain()
    assert seen == ["Solve for x: 5x - 2 = 13."]
    text = page_context(s, coach, matched=lambda texts: m.cached(texts, s.laptop))
    assert "Assignment (PS4)" in text and "[3] A 5 kg block" in text and "answering assignment question 2" in text and "Work A:" in text
