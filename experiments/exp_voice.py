"""Voice loop test without a microphone: synthesized speech -> speech detection -> Parakeet -> wake phrase ->
Nemotron answering from a saved page. Times every stage.

    uv run experiments/exp_voice.py [--session experiments/data/demo_session.json] [--speak]
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from margin.coach import Coach
from margin.converse import Conversation, is_generic_help, page_context
from margin.nebius import Nebius
from margin.session import Session
from margin.voice import FRAME, SAMPLE_RATE, Segmenter, Transcriber, parse_wake

ROOT = Path(__file__).resolve().parent
UTTERANCES = [
    "Hey Margin, can you help me with this problem?",
    "Hey Margin, why do I add two to both sides in the second problem?",
    "Hey Margin, wait, why?",
    "Hey Margin, is my first problem right?",
    "I think I need to buy groceries later today.",
    "Hey Margin, what does it mean that there are two answers?",
]


def synthesize(text: str, path: Path) -> np.ndarray:
    subprocess.run(["say", "-o", str(path), "--data-format=LEI16@16000", text], check=True)
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SAMPLE_RATE
    lead, tail = np.zeros(int(0.6 * SAMPLE_RATE), np.float32), np.zeros(int(1.2 * SAMPLE_RATE), np.float32)
    rng = np.random.default_rng(0)
    room = lambda a: a + 0.002 * rng.standard_normal(len(a)).astype(np.float32)  # quiet room hiss
    return np.concatenate([room(lead), room(audio), room(tail)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=str(ROOT / "data" / "demo_session.json"))
    ap.add_argument("--speak", action="store_true", help="also say the replies out loud")
    args = ap.parse_args()

    session = Session.from_dict(json.loads(Path(args.session).read_text()))
    client = Nebius()
    coach = Coach(session, check_fn=lambda *a: None, help_fn=lambda *a: None)
    last = max(session.blocks.values(), key=lambda b: b.bbox[1])
    coach.focus = (last.id, last.problems()[-1])
    conversation = Conversation(client)

    t = time.perf_counter()
    asr = Transcriber()
    print(f"Parakeet ready in {time.perf_counter() - t:.1f}s\n")
    tmp = Path(tempfile.mkdtemp())
    rows = []

    for i, text in enumerate(UTTERANCES):
        audio = synthesize(text, tmp / f"u{i}.wav")
        seg = Segmenter()
        clock, utterance, spoke_until, detected_at = 0.0, None, 0.0, 0.0
        for start in range(0, len(audio) - FRAME + 1, FRAME):
            clock = start / SAMPLE_RATE
            out = seg.push(audio[start : start + FRAME], clock)
            if out is not None:
                utterance, spoke_until, detected_at = out, seg.last_voice_time, clock + FRAME / SAMPLE_RATE
                break
        if utterance is None:
            print(f"!! no speech detected for: {text}")
            continue
        endpoint = detected_at - spoke_until

        t0 = time.perf_counter()
        heard = asr.transcribe(utterance)
        asr_s = time.perf_counter() - t0
        question = parse_wake(heard)
        print(f"🎙  said:  {text}\n    heard: {heard}   (ASR {asr_s * 1000:.0f} ms)")
        if question is None:
            print("    -> no wake phrase: ignored, nothing sent\n")
            continue

        t1 = time.perf_counter()
        first, parts = None, []
        found = coach.take_focus_mistake() if is_generic_help(question) else None
        replies = iter([found.message]) if found else conversation.reply(question, page_context(session, coach), session.course)
        for sentence in replies:
            if first is None:
                first = time.perf_counter() - t1
            parts.append(sentence)
        total_first = endpoint + asr_s + first
        reply = " ".join(parts)
        rows.append((endpoint, asr_s, first, total_first))
        print(f"    Margin: {reply}")
        print(f"    ⏱ end-of-speech wait {endpoint:.2f}s + ASR {asr_s:.2f}s + first sentence {first:.2f}s = {total_first:.2f}s\n")
        if args.speak:
            subprocess.run(["say", reply])

    if rows:
        cols = np.array(rows)
        med = np.median(cols, axis=0)
        print("--- median latency (before text-to-speech) ---")
        print(f"end-of-speech wait {med[0]:.2f}s | Parakeet {med[1] * 1000:.0f} ms | Nemotron first sentence {med[2]:.2f}s | total {med[3]:.2f}s")


if __name__ == "__main__":
    main()
