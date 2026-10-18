"""Voice in and out, all on the Mac.

- Microphone audio is split into utterances by a small energy-based speech detector (`Segmenter`).
- Every utterance is transcribed locally by NVIDIA Parakeet (MLX, on the GPU) - nothing leaves the Mac
  until it's a question for Margin.
- "Hey Margin, ..." is detected in the transcript (`parse_wake`), so there's no wake-word model to train.
- Tapping Enter (or holding a key) skips the wake phrase: the next thing you say is the question.
- `Speaker` speaks sentence by sentence as they stream in, and can be interrupted instantly.
"""

import difflib
import queue
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME = 480  # 30 ms

# --------------------------------------------------------------------------------------------- input


class Segmenter:
    """Turns a stream of 30 ms frames into utterances. The noise floor adapts, so it works in a quiet room
    and a noisy library alike."""

    def __init__(
        self,
        start_db: float = 12.0,
        end_silence_ms: int = 700,
        min_speech_ms: int = 250,
        max_ms: int = 15000,
        preroll_ms: int = 300,
    ):
        self.start_db = start_db
        self.end_frames = end_silence_ms // 30
        self.min_frames = min_speech_ms // 30
        self.max_frames = max_ms // 30
        self.preroll = preroll_ms // 30
        self.floor = -60.0
        self._calibration: list[float] | None = []  # first ~0.5 s measures the room
        self._buf: list[np.ndarray] = []
        self._recent: list[np.ndarray] = []
        self._speaking = False
        self._voiced = 0
        self._silent = 0
        self.last_voice_time = 0.0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def voiced_frames(self) -> int:
        """Voiced frames in the utterance in progress (0 when not in speech)."""
        return self._voiced if self._speaking else 0

    def partial(self, last_frames: int | None = None) -> np.ndarray:
        """Audio of the utterance in progress so far (or just its last `last_frames` frames)."""
        buf = self._buf[-last_frames:] if last_frames else self._buf
        return np.concatenate(buf) if buf else np.zeros(0, dtype=np.float32)

    def trim(self, keep_frames: int) -> None:
        """Drop the start of the utterance in progress (e.g. Margin's own voice before the student cut in)."""
        if len(self._buf) > keep_frames:
            self._buf = self._buf[-keep_frames:]
            self._voiced = min(self._voiced, keep_frames)

    def push(self, frame: np.ndarray, now: float | None = None) -> np.ndarray | None:
        """Feed one frame; returns a finished utterance (float32 audio) or None."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-10)
        db = 20 * np.log10(rms)
        if self._calibration is not None:
            self._calibration.append(db)
            if len(self._calibration) >= 15:
                self.floor = max(-75.0, float(np.median(self._calibration)))
                self._calibration = None
            return None
        voiced = db > self.floor + self.start_db
        if not self._speaking:
            self.floor = max(-75.0, 0.95 * self.floor + 0.05 * db) if not voiced else self.floor
            self._recent = (self._recent + [frame])[-self.preroll :]
            if voiced:
                self._speaking, self._buf, self._voiced, self._silent = True, list(self._recent), 1, 0
                self.last_voice_time = now if now is not None else time.monotonic()
            return None
        self._buf.append(frame)
        if voiced:
            self._voiced += 1
            self._silent = 0
            self.last_voice_time = now if now is not None else time.monotonic()
        else:
            self._silent += 1
        if self._silent >= self.end_frames or len(self._buf) >= self.max_frames:
            audio = np.concatenate(self._buf)
            enough = self._voiced >= self.min_frames
            self._speaking, self._buf, self._recent = False, [], []
            return audio if enough else None
        return None


_WAKE_GREETINGS = {"hey", "hi", "ok", "okay", "yo", "hello"}


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUMBER_WORDS = {w: str(i) for i, w in enumerate(_ONES)} | {w: str(i * 10) for i, w in enumerate(_TENS) if w != "_"}


# Words text-to-speech produces from math ("5x - 2 = 13" is read "five x minus two equals thirteen") and that
# speech recognition then garbles ("five times"); they say nothing about who is talking.
_MATH_WORDS = {"times", "equals", "equal", "plus", "minus", "squared", "cubed", "over", "divided", "root", "power", "negative"}
# How people cut in on someone who is talking.
_INTERRUPT_CUES = {"wait", "hold", "hang", "stop", "but", "actually", "sorry", "excuse", "no", "nope", "hey", "margin",
                   "what", "why", "how", "question", "oh"}
_STOP_CUES = {"wait", "stop", "hold", "hang"}


def _words(text: str) -> list[tuple[str, int]]:
    """Lowercase words with start offsets; number words become digits ("sixteen" -> "16") and "5x" -> "5", "x"."""
    return [
        (_NUMBER_WORDS.get(m.group().lower(), m.group().lower()), m.start())
        for m in re.finditer(r"[A-Za-z']+|\d+(?:\.\d+)?", text)
    ]


def _content(w: str) -> bool:
    return len(w) >= 3 and not w[0].isdigit() and w not in _MATH_WORDS


def strip_echo(heard: str, spoken: str, interrupting: bool = False) -> str:
    """Remove Margin's own words (speaker leaking into the mic) from what was heard; return what's left of the
    student's speech, or "" if it was all echo. Handles a question glued onto the end of Margin's sentence:
    "...x squared equals sixteen. What about problem two?" -> "What about problem two?"

    Only content words are judged (3+ letters, not numbers or spoken-math words, which match too much or get
    garbled). A word counts as Margin's if Margin said it or a variant of it ("move" / "moved").

    interrupting=True (Margin is mid-sentence) asks for more: the student's part must contain a cue people use
    to cut in ("wait", "hold on", "but", "why"...) or be 4+ clearly-not-Margin words. A lone "wait"/"stop" is
    enough to stop it."""
    spoken_words = {w for w, _ in _words(spoken) if _content(w)}
    all_heard = _words(heard)
    words = [(w, at) for w, at in all_heard if _content(w)]

    def margins(w: str) -> bool:
        return w in spoken_words or any(sw.startswith(w) or w.startswith(sw) for sw in spoken_words)

    def accept(start: int, theirs: int) -> str:
        tail = heard[start:].strip()
        if interrupting:
            cues = {w for w, at in all_heard if at >= start} & _INTERRUPT_CUES
            if not (cues - spoken_words) and theirs < 4:
                return ""
        return tail

    if not spoken_words:
        return heard.strip() if (words or not interrupting) else ""
    echo = [margins(w) for w, _ in words]
    # The student's part is the longest tail that is mostly not Margin's words and starts with one of theirs.
    for i in range(len(words)):
        tail = echo[i:]
        theirs = tail.count(False)
        if not echo[i] and theirs >= 2 and theirs / len(tail) >= 0.6:
            return accept(words[i][1], theirs)
    if interrupting:
        for w, at in all_heard:
            if w in _STOP_CUES and w not in spoken_words:
                return heard[at:].strip()  # "wait!" / "stop" on its own
        return ""
    if not words or sum(echo) / len(echo) >= 0.5:
        return ""
    return heard.strip()


def is_echo(heard: str, spoken: str) -> bool:
    """True if what the mic heard is entirely Margin's own words."""
    return bool(heard.strip()) and not strip_echo(heard, spoken)


def parse_wake(transcript: str) -> str | None:
    """If the utterance starts with "hey Margin" (or close), return what follows it ("" if nothing).
    Otherwise None. Fuzzy, because speech recognition hears "Margin" as "margen", "marjin", ..."""
    words = re.findall(r"[a-z']+", transcript.lower())
    for i, w in enumerate(words[:3]):
        if difflib.SequenceMatcher(None, w, "margin").ratio() >= 0.72:
            if i == 0 or all(p in _WAKE_GREETINGS for p in words[:i]):
                rest = re.split(r"(?i)\b" + re.escape(w) + r"\b[,.!?]*", transcript, maxsplit=1)
                return rest[1].strip() if len(rest) > 1 else ""
    return None


class Transcriber:
    """NVIDIA Parakeet speech-to-text, running locally on Apple silicon via MLX."""

    def __init__(self, model_id: str = "mlx-community/parakeet-tdt-0.6b-v3"):
        import soundfile  # noqa: F401  (fail early if missing)
        from parakeet_mlx import from_pretrained

        self.model = from_pretrained(model_id)
        self._dir = Path(tempfile.mkdtemp(prefix="margin-asr-"))
        self._lock = threading.Lock()
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))  # first call compiles kernels; do it now

    def transcribe(self, audio: np.ndarray) -> str:
        import soundfile as sf

        with self._lock:
            path = self._dir / "utterance.wav"
            sf.write(path, audio, SAMPLE_RATE)
            return self.model.transcribe(path).text.strip()


@dataclass
class Question:
    text: str
    spoke_until: float  # when the student stopped talking (monotonic), for latency measurement
    asr_s: float
    via: str  # "wake" | "tap" | "hold"


class VoiceInput:
    """Listens on the mic in the background and puts `Question`s on `questions`.

    - "Hey Margin, <question>" (or "Hey Margin" ... <question>) any time.
    - `arm()`: the next utterance is a question, no wake phrase needed (Enter key, and the follow-up window
      right after Margin answers, so a conversation flows).
    - Barge-in: while Margin is speaking, if the student starts talking (and it isn't Margin's own voice
      leaking from the speakers), Margin stops within ~half a second and what they say is the question.
    - `hold_start()` / `hold_end()`: everything said while a key is held is the question.
    """

    BARGE_IN_FRAMES = 15  # re-check every ~450 ms of speech heard while Margin talks
    BARGE_IN_WINDOW = 67  # ...looking at the last ~2 s
    BARGE_IN_KEEP = 50  # once the student cuts in, keep only the last ~1.5 s (drop Margin's voice before it)

    def __init__(
        self, transcriber, on_wake=None, armed_timeout: float = 6.0, is_speaking=None, spoken_text=None, debug=False
    ):
        self.transcriber = transcriber
        self.on_wake = on_wake or (lambda: None)  # called the moment we know the student is talking to us
        self.is_speaking = is_speaking or (lambda: False)
        self.spoken_text = spoken_text or (lambda: "")  # what Margin said recently, for echo rejection
        self.questions: queue.Queue[Question] = queue.Queue()
        self.armed_timeout = armed_timeout
        self._frames: queue.Queue[np.ndarray] = queue.Queue()
        self._seg = Segmenter()
        self._armed_until = 0.0
        self._armed_via = "tap"
        self._hold: list[np.ndarray] | None = None
        self._barged = False  # already decided the student cut in during this utterance
        self._checked_at = 0  # voiced-frame count at the last barge-in check
        self.debug = debug
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME, callback=self._callback
        )
        self._stream.start()
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()

    def arm(self, via: str = "tap", seconds: float | None = None) -> None:
        self._armed_until = time.monotonic() + (self.armed_timeout if seconds is None else seconds)
        self._armed_via = via

    @property
    def armed(self) -> bool:
        return time.monotonic() <= self._armed_until

    def hold_start(self) -> None:
        self._hold = []

    def hold_end(self) -> None:
        frames, self._hold = self._hold, None
        if frames:
            self._emit(np.concatenate(frames), time.monotonic(), "hold")

    def _callback(self, indata, frames, t, status) -> None:
        self._frames.put(indata[:, 0].copy())

    def _run(self) -> None:
        while True:
            self.process(self._frames.get())

    def process(self, frame: np.ndarray, now: float | None = None) -> None:
        """Handle one 30 ms frame of mic audio."""
        if self._hold is not None:
            self._hold.append(frame)
            return
        utterance = self._seg.push(frame, now)
        # While Margin talks, keep re-checking: with laptop speakers the mic hears Margin the whole time, so
        # the student's voice shows up *inside* an utterance that started as Margin's own echo.
        voiced = self._seg.voiced_frames
        if (
            self._seg.speaking
            and not self._barged
            and voiced - self._checked_at >= self.BARGE_IN_FRAMES
            and self.is_speaking()
        ):
            self._checked_at = voiced
            self._check_barge_in()
        if utterance is not None or not self._seg.speaking:
            self._barged, self._checked_at = False, 0
        if utterance is not None:
            self._handle(utterance, self._seg.last_voice_time)

    def _check_barge_in(self) -> None:
        """Margin is talking and the mic hears speech: is it the student, or Margin's own voice?"""
        text = self.transcriber.transcribe(self._seg.partial(self.BARGE_IN_WINDOW))
        theirs = strip_echo(text, self.spoken_text(), interrupting=True)
        if self.debug:
            print(f"   [voice] while speaking, heard {text!r} -> " + (f"YOU: {theirs!r}, interrupting" if theirs else "echo, ignored"))
        if theirs:
            self._barged = True
            self.on_wake()  # stop talking right away
            self._seg.trim(self.BARGE_IN_KEEP)  # their question starts here, not with Margin's echo
            self.arm("barge-in")  # the rest of this utterance is their question

    def _handle(self, audio: np.ndarray, spoke_until: float) -> None:
        armed = time.monotonic() <= self._armed_until + 1.0
        t0 = time.monotonic()
        text = self.transcriber.transcribe(audio)
        asr = time.monotonic() - t0
        theirs = strip_echo(text, self.spoken_text()) if text else ""
        if self.debug:
            note = " (echo, ignored)" if text and not theirs else (f" -> kept {theirs!r}" if theirs != text else "")
            print(f"   [voice] heard {text!r}{note}{' (armed)' if armed else ''}")
        if not theirs:
            return  # silence, or Margin hearing itself
        text = theirs
        rest = parse_wake(text)
        if armed:
            self._armed_until = 0.0
            if rest is not None:
                self.on_wake()
            self.questions.put(Question(rest if rest else text, spoke_until, asr, self._armed_via))
            return
        if rest is None:
            return  # not for us; nothing leaves the Mac
        self.on_wake()
        if rest:
            self.questions.put(Question(rest, spoke_until, asr, "wake"))
        else:
            chime()
            self.arm("wake")  # "hey Margin" ... (pause) ... question

    def _emit(self, audio: np.ndarray, spoke_until: float, via: str) -> None:
        t0 = time.monotonic()
        text = self.transcriber.transcribe(audio)
        if text:
            rest = parse_wake(text)
            self.questions.put(Question(rest if rest else text, spoke_until, time.monotonic() - t0, via))


# -------------------------------------------------------------------------------------------- output


def chime() -> None:
    subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"])


class Speaker:
    """Speaks queued sentences one after another (macOS `say`); `stop()` cuts it off immediately."""

    def __init__(self, voice: str | None = None, rate: int | None = None, dry_run: bool = False):
        self.voice, self.rate, self.dry_run = voice, rate, dry_run
        self._q: queue.Queue[tuple[int, str]] = queue.Queue()
        self._gen = 0
        self._proc: subprocess.Popen | None = None
        self.first_audio_at: float | None = None
        self._recent: list[tuple[float, str]] = []  # (when queued, text): Margin's own words, for echo rejection
        self.finished_at = 0.0  # when the last sentence finished playing
        threading.Thread(target=self._run, daemon=True).start()

    def recent_text(self, seconds: float = 20.0) -> str:
        cutoff = time.monotonic() - seconds
        return " ".join(text for t, text in self._recent if t >= cutoff)

    @property
    def speaking(self) -> bool:
        return not self._q.empty() or (self._proc is not None and self._proc.poll() is None)

    def say(self, text: str) -> None:
        text = text.strip()
        if text:
            self._recent = [(t, x) for t, x in self._recent if t >= time.monotonic() - 60] + [(time.monotonic(), text)]
            self._q.put((self._gen, text))

    def stop(self) -> None:
        self._gen += 1
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self) -> None:
        while True:
            gen, text = self._q.get()
            if gen != self._gen:
                continue  # interrupted
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            if self.dry_run:
                self.finished_at = time.monotonic()
                continue
            cmd = ["say"]
            if self.voice:
                cmd += ["-v", self.voice]
            if self.rate:
                cmd += ["-r", str(self.rate)]
            self._proc = subprocess.Popen(cmd + [text])
            self._proc.wait()
            self.finished_at = time.monotonic()
