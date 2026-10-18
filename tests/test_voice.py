import numpy as np

from margin.coach import Coach
from margin.converse import (Conversation, is_generic_help, is_goodbye, page_changes, page_context, readable,
                             sentences, speakable, wants_look)
from margin.session import Session
from margin.transcribe import Line, Transcription
from margin.voice import FRAME, SAMPLE_RATE, Segmenter, VoiceInput, is_echo, parse_wake, strip_echo


def frames(audio: np.ndarray):
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        yield audio[i : i + FRAME]


def tone(seconds: float, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds: float, amp: float = 0.002, seed: int = 0) -> np.ndarray:
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * SAMPLE_RATE))).astype(np.float32)


def run(seg: Segmenter, audio: np.ndarray) -> list[np.ndarray]:
    return [u for f in frames(audio) if (u := seg.push(f, 0.0)) is not None]


def test_segmenter_finds_one_utterance_between_silences():
    utts = run(Segmenter(), np.concatenate([quiet(1.0), tone(1.2), quiet(1.5, seed=1)]))
    assert len(utts) == 1
    assert 1.2 <= len(utts[0]) / SAMPLE_RATE <= 2.3  # speech + preroll + trailing silence


def test_segmenter_splits_two_utterances_and_ignores_clicks():
    audio = np.concatenate([quiet(1.0), tone(0.06), quiet(1.0, seed=2), tone(0.8), quiet(1.0, seed=3), tone(0.8), quiet(1.0, seed=4)])
    assert len(run(Segmenter(), audio)) == 2  # the 60 ms click is too short to be speech


def test_segmenter_adapts_to_a_noisy_room():
    noisy = 0.03  # loud background
    audio = np.concatenate([quiet(2.0, noisy), tone(1.0, 0.3) + quiet(1.0, noisy, 5), quiet(1.5, noisy, 6)])
    assert len(run(Segmenter(), audio)) == 1


def test_parse_wake():
    assert parse_wake("Hey Margin, can you help me with this problem?") == "can you help me with this problem?"
    assert parse_wake("hey margen why do I subtract two") == "why do I subtract two"
    assert parse_wake("Margin, is this right?") == "is this right?"
    assert parse_wake("Hey Margin.") == ""
    assert parse_wake("I think I need to buy groceries") is None
    assert parse_wake("the margin of error is small") is None  # "margin" mid-sentence isn't a wake word


def test_generic_help_routing():
    for q in ["can you help me?", "Can you help me with this problem?", "am I right", "is this correct?", "help", "I'm stuck"]:
        assert is_generic_help(q), q
    for q in ["why do I subtract two from both sides?", "what does entropy mean"]:
        assert not is_generic_help(q), q


def test_sentences_stream_early_and_keep_decimals_intact():
    pieces = ["Close! Look at your sec", "ond line. What is 9.", "8 times two? Then", " try again"]
    assert list(sentences(iter(pieces))) == ["Close! Look at your second line.", "What is 9.8 times two?", "Then try again"]


def test_page_context_marks_focus_circles_and_found_mistakes():
    s = Session()
    lines = [Line("Solve 5x - 2 = 13", None, None, "math", True, 0.9), Line("5x = 11", None, None, "math", True, 0.9)]
    block = s.apply((0, 0, 100, 100), Transcription(lines, True, 1, raw=None))
    block.flags.append({"line": "5x = 11", "reason": "added 2 wrong", "hint": "h"})
    coach = Coach(s, check_fn=lambda *a: None, help_fn=lambda *a: None)
    coach.focus = (block.id, block.problems()[0])
    text = page_context(s, coach)
    assert "working on this now" in text and "(circled a ?)" in text and 'Already found: "5x = 11"' in text
    assert "nothing written" in page_context(Session())


def test_speakable_math():
    assert speakable("You have x² = 16 and x^2 - 9 = 0.") == "You have ex squared equals 16 and ex squared minus 9 equals 0."
    assert speakable("Undo the -2: what is 13 + 2?") == "Undo the minus 2: what is 13 + 2?"
    assert speakable("so x = ±4") == "so ex equals plus or minus 4"
    assert speakable("2 × 9.8 ÷ 4") == "2 times 9.8 divided by 4"
    assert speakable("a well-known step-by-step method") == "a well-known step-by-step method"
    assert speakable("(-4)² is also 16") == "minus 4 squared is also 16"
    assert speakable("You found one of them – can you think of the other?") == "You found one of them, can you think of the other?"
    assert speakable("so x - 2 = 13 and 16 - x") == "so ex minus 2 equals 13 and 16 minus ex"
    assert speakable("You wrote 5x = 11.") == "You wrote 5 ex equals 11."  # not "five times"
    assert speakable("a 5kg block, 3y + 2") == "a 5kg block, 3 y + 2"  # units untouched; variables split
    assert readable("You wrote 5x = 11.") == "You wrote 5x = 11."  # the screen keeps the math as written
    assert speakable(r"it's sine: \( F_{\parallel} = mg \sin\theta \).") == "it's sine: F parallel equals mg sin theta ."
    assert speakable(r"\frac{1}{2} a t^2") == "1 over 2 a t squared"


class FakeASR:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def transcribe(self, audio):
        self.calls += 1
        return self.text


def listen(text: str, audio: np.ndarray, speaking: bool = False, spoken: str = "", armed: str | None = None):
    events = []
    vi = VoiceInput(FakeASR(text), on_wake=lambda: events.append("stop"),
                    is_speaking=lambda: speaking, spoken_text=lambda: spoken)
    if armed:
        vi.arm(armed)
    for i, f in enumerate(frames(audio)):
        vi.process(f, i * 0.03)
        if events == ["stop"] and "stopped_at" not in locals():
            stopped_at = i * 0.03
    questions = []
    while not vi.questions.empty():
        questions.append(vi.questions.get())
    return events, questions, locals().get("stopped_at")


SPEECH = np.concatenate([quiet(1.0), tone(1.5), quiet(1.2, seed=9)])


def test_is_echo():
    said = "Look at your second line. What do you do to both sides to cancel the minus two?"
    assert is_echo("what do you do to both sides", said)
    assert not is_echo("wait why do I add two", said)
    assert not is_echo("anything", "")


def test_follow_up_needs_no_wake_phrase():
    _, qs, _ = listen("wait, why?", SPEECH, armed="follow-up")
    assert [(q.text, q.via) for q in qs] == [("wait, why?", "follow-up")]


def test_without_wake_or_window_speech_is_ignored():
    _, qs, _ = listen("wait, why?", SPEECH)
    assert qs == []


def test_barge_in_stops_margin_early_and_takes_the_question():
    events, qs, stopped_at = listen("hold on, why minus two", SPEECH, speaking=True,
                                    spoken="You're trying to get x alone. What's attached to the 5x?")
    assert events and events[0] == "stop"
    assert stopped_at is not None and stopped_at < 1.0 + 0.6  # mid-utterance, well before they finish
    assert [q.via for q in qs] == ["barge-in"]


def test_margins_own_voice_does_not_interrupt_it():
    said = "You're trying to get x alone. What's attached to the 5x?"
    events, qs, _ = listen("you're trying to get x alone", SPEECH, speaking=True, spoken=said, armed="follow-up")
    assert events == [] and qs == []


def test_barge_in_through_continuous_speaker_echo():
    """Laptop speakers: the mic hears Margin the whole time, then the student talks over it."""
    said = "When you take the square root remember both the positive and negative roots what could the other one be"
    echo = tone(3.0)  # Margin's voice leaking in (220 Hz)
    t = np.arange(int(1.5 * SAMPLE_RATE)) / SAMPLE_RATE
    student = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # the student (440 Hz), over the tail of it
    audio = np.concatenate([quiet(1.0), echo, student, quiet(1.2, seed=11)])

    class ASR:
        def transcribe(self, a):
            tail = a[-int(0.3 * SAMPLE_RATE):]
            crossings = np.sum(np.abs(np.diff(np.sign(tail)))) / 2 / 0.3
            return "hold on what about problem two" if crossings > 600 else "remember both the positive and negative roots"

    events, now = [], {"speaking": True}
    vi = VoiceInput(ASR(), on_wake=lambda: (events.append("stop"), now.update(speaking=False)),
                    is_speaking=lambda: now["speaking"], spoken_text=lambda: said)
    stopped_at = None
    for i, f in enumerate(frames(audio)):
        vi.process(f, i * 0.03)
        if events and stopped_at is None:
            stopped_at = i * 0.03
    questions = []
    while not vi.questions.empty():
        questions.append(vi.questions.get())
    assert stopped_at is not None and 4.0 <= stopped_at <= 4.0 + 1.0  # within ~1 s of the student starting
    assert [(q.text, q.via) for q in questions] == [("hold on what about problem two", "barge-in")]


def test_goodbye_detection():
    for q in ["Okay, thank you, bye.", "thanks!", "that's all", "I'm good, thanks", "never mind"]:
        assert is_goodbye(q), q
    for q in ["thanks, but why is it negative?", "so the two solutions are plus and minus four", "wait, why?"]:
        assert not is_goodbye(q), q


SAID = "You're on the right track! You correctly moved the 16 over to get x squared equals 16."


def test_strip_echo_on_the_real_log():
    # Exactly what the mic heard in the live test, with Margin's speech as spoken.
    assert strip_echo("You correctly move the sixteen over to get x squared equals sixteen. What about problem two?", SAID) \
        == "What about problem two?"
    for echo in ["You're on the right side.", "You're on the right track.", "You correctly move the circuit.",
                 "You correctly move the sixteen over to", "Move the sixteen over to get X squared."]:
        assert strip_echo(echo, SAID) == "", echo
    assert strip_echo("What about problem two?", SAID) == "What about problem two?"
    assert strip_echo("wait why", "") == "wait why"


def test_spoken_math_echo_does_not_interrupt():
    # From the second live test: Margin's own math, garbled by the mic, was mistaken for the student.
    said = "You added 2 to 13 and got 11? Let me check your work: 5x minus 2 equals 13, so you need to add 2 to both sides."
    for garbled in ["Five times.", "Five X equals one.", "You have five times each other", "You have five times equal."]:
        assert strip_echo(garbled, said, interrupting=True) == "", garbled
    said2 = "Five x equals what? You have 5x equals 11, but if you add 2 to 13, you should get 15."
    assert strip_echo("You have five times each other", said2, interrupting=True) == ""


def test_real_interruptions_still_work():
    said = "You're on the right track with x squared equals 16. Remember, when you take the square root of both sides"
    assert strip_echo("Wait, what about problem two?", said, interrupting=True) == "Wait, what about problem two?"
    assert strip_echo("With X squared equals six. Hold on, why is that?", said, interrupting=True) == "Hold on, why is that?"
    assert strip_echo("wait", said, interrupting=True) == "wait"
    assert strip_echo("can you explain the second problem again please", said, interrupting=True) != ""  # 4+ words, no cue
    assert strip_echo("Right track with X squared equals six.", said, interrupting=True) == ""


def test_wants_look():
    for q in ["look at my laptop", "can you look at my screen", "check the assignment on my computer", "read my problem set"]:
        assert wants_look(q), q
    for q in ["look at line two", "can you help me?", "why do I add two"]:
        assert not wants_look(q), q


BEFORE = "Work A:\n  line 1: Solve 5x - 2 = 13\n  line 2: 5x = 11"
AFTER = "Work A:\n  line 1: Solve 5x - 2 = 13\n  line 2: 5x = 15\n  line 3: x = 3  (circled a ?)"


def test_page_changes():
    assert page_changes(None, AFTER) == []
    assert page_changes(BEFORE, AFTER) == ['changed "5x = 11" to "5x = 15"', 'wrote "x = 3"']
    assert page_changes(AFTER, BEFORE) == ['changed "5x = 15" to "5x = 11"', 'erased "x = 3"']


def test_current_page_sits_next_to_the_newest_question():
    c = Conversation(client=None)
    c._last_page = BEFORE
    c.remember("is this right?", "Look at your second line: 5x equals 11.")
    msgs = c.messages("is it right now?", AFTER)
    assert "5x = 15" not in msgs[0]["content"]  # not buried in the system prompt
    last = msgs[-1]["content"]
    assert "RIGHT NOW" in last and "5x = 15" in last and 'changed "5x = 11" to "5x = 15"' in last
    assert last.endswith("[They said] is it right now?")
