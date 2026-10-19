# Margin

A study companion that reads your iPad handwriting as you write, knows the problem on your laptop, and speaks up the moment your reasoning slips. See [SPEC.md](SPEC.md) for the full design.

Built for the Nebius x NVIDIA Global AI Hackathon on NVIDIA Nemotron via Nebius Token Factory.

## Status: day 1–2 de-risking

| Piece | Where | State |
|---|---|---|
| iPad USB capture (no iPad app) | `capture/` (Swift) | **works live** — 1200×1600 @ 4 fps over USB |
| Ink tracker (new ink / erase / zoom / page turn / UI filtering) | `margin/ink.py` | **verified on a live GoodNotes recording** |
| Handwriting → lines (Nemotron 3.5 Super-VL) | `margin/transcribe.py` | **5/5 real pages read correctly**, ~1.7 s/page |
| Exact algebra checking (SymPy) | `margin/checker/algebra.py` | tested |
| Checker cascade (SymPy → Lightning → Super) | `margin/checker/cascade.py` | **9/9 caught, 0/7 false alarms** on planted set |
| Live session state (blocks merged across re-reads) | `margin/session.py` | verified on live recording |
| Coach (auto-check, circled-? help, quiet period, no repeats) | `margin/coach.py`, `margin/tutor.py` | verified on live recording |
| Live runner (+ spoken alerts via macOS `say`) | `margin/live.py` | replay verified; ready for live use |
| Assignment: dropped PDF (read on the Mac) or "look at my laptop" (one screenshot, on request) | `margin/assignment.py`, `margin/laptop.py` | iPad work matched to questions; checker + voice use the real question |
| On-screen overlay: captions, status, the line Margin means, "what I read" | `margin/overlay.py`, `margin/overlay_ui/`, `overlay-app/` (floating window) | verified on a replayed live session |
| Voice loop ("hey Margin" / Enter / hold key → Parakeet → Nemotron → speech) | `margin/voice.py`, `margin/converse.py` | **1.36 s median to first sentence** (synthetic speech) |

## Setup

```bash
uv sync
cp .env.example .env        # then paste your NEBIUS_API_KEY
uv run scripts/list_models.py   # copy the real Nemotron model IDs into .env
```

Build the capture tool (needs Xcode):

```bash
cd capture && swift build -c release && cd ..
```

## Run Margin live

Terminal 1 (iPad plugged in, GoodNotes open):

```bash
capture/.build/release/ipad-capture --out frames --fps 4
```

Terminal 2:

```bash
uv run python -m margin.live --voice
```

Talk to it: **"Hey Margin, …"** any time, or **tap Enter** and speak, or add `--hold-key alt_r` and **hold Right Option**
while you speak (needs Input Monitoring permission for Terminal). The first run asks for Microphone access and
downloads Parakeet (~600 MB). Speech recognition runs on your Mac — nothing is sent until you say "hey Margin".
**Overlay:** Margin serves a live overlay at http://127.0.0.1:8765/ (local only). For a floating always-on-top window,
build it once (`cd overlay-app && swift build -c release`) and add `--panel`. It shows what Margin heard and said, the
problem you're on with its assignment question, the line Margin is talking about (with the hint; "Show why" for the
full reason), and **What I read**: your ink next to how Margin read it. Preview it without an iPad or mic:
`uv run python -m margin.overlay_demo` and open http://127.0.0.1:8765/.

**Give Margin the assignment:** start with `--problems ps4.pdf`, or **drag the PDF onto the Margin terminal and press
Enter**. A normal PDF is read on your Mac and never uploaded; a scanned one is sent to Nemotron once to read it.
Or say **"Hey Margin, look at my laptop"** (or type `look`): Margin takes *one* screenshot of the front-most browser /
PDF / docs window — only then, never in the background, and never of other apps (Messages, Mail...).
Each problem on your iPad is matched to the question it answers, so checks and answers use the real question.

Try voice without the iPad: `uv run python -m margin.live --voice --talk-only experiments/data/demo_session.json`.

**Ask mode (default):** Margin checks everything silently in the background and only speaks when you ask —
press **Enter** in the Margin terminal (stand-in for "hey Margin, can you help?") or draw a **circled ?**.
Because the checking already happened, the answer is instant and points at the exact line.
**Coach mode** (`--mode coach`) also speaks up about mistakes unprompted.

Write normally. Margin re-reads each block of writing when you pause, splits it into problems, checks each
problem in the background, and answers a circled "?" right away. Unprompted mistake alerts wait a short grace
period (so a half-written line isn't called wrong), come at most once per line, and at most once per
`--quiet-seconds` (default 20). Memory is keyed by content, so zooming/re-reading never repeats an alert, and
writing cut off by the screen edge is shown but never judged. Live state goes to `frames/session.json`.

Replay a recording: `uv run python -m margin.live --replay frames/rec` (add `--realtime` for faithful timing).

Live test session (2026-10-17, 6 scripted tests incl. 4 zooms): 2 alerts, both correct (`5x = 11` caught
unprompted; circled ? on `x = 4` answered once), 0 false alarms, 0 repeats.

## Day-1 tests

**1. iPad capture.** Plug in the iPad, unlock it, tap Trust. Run this in **Terminal** (it asks for Camera permission the first time, because macOS treats the iPad screen as a camera):

```bash
capture/.build/release/ipad-capture --out frames --fps 4 --record
```

`frames/latest.jpg` updates live; `frames/rec/` keeps every frame for replay.

**2. Ink tracking.** With capture running, in a second terminal, write in GoodNotes/Notability and watch regions appear:

```bash
uv run experiments/watch_ink.py --frames frames
```

Add `--transcribe` to send each new region to Nemotron 3.5 Super-VL. Replay a recording with `--replay frames/rec`. Crops land in `experiments/results/crops/`.

**3. Handwriting accuracy.** Drop ~10 screenshots of your real notes into `experiments/notes/` (optionally with a same-named `.txt` of what you actually wrote, for an error rate), then:

```bash
uv run experiments/exp_handwriting.py
```

**4. Mistake checking.** Offline (SymPy only) and full cascade:

```bash
uv run experiments/exp_checker.py --offline
uv run experiments/exp_checker.py
```

Results on the 16 planted problems (2026-10-17):

| Layer | Caught | False alarms | Latency (median) |
|---|---|---|---|
| SymPy only | 6/9 | 0/7 | ~0 ms, no AI calls |
| SymPy → Nemotron 3.5 Lightning → Nemotron 3 Super | **9/9** | **0/7** | 2.0 s |

SymPy misses the semantic errors (sin/cos swap, missing chain rule, inverted mole ratio); the Nemotron layers catch them. On one correct problem Lightning raised a false alarm that Super correctly overruled.

**University level** (`--data experiments/data/university_errors.json`; 16 problems across calculus, integration by
parts, eigenvalues, Lagrangian optimization, probability, the Master Theorem, ODEs and series; 2026-10-19):

| Configuration | Caught (right line) | False alarms | Median |
|---|---|---|---|
| SymPy only | 1/8 | 0/8 | ~0 s |
| Lightning screen → Super confirm | 6/8 | 0/8 | 1.9 s |
| Lightning screen → **Ultra** confirm | 5/8 | 0/8 | 2.6 s |
| **Super screen → Super confirm** (default) | **7/8** | **0/8** | 2.2 s |

At this level the fast screen was the bottleneck (it never flagged the misses), so screening now runs on Nemotron 3
Super; it's in the background, so the extra time isn't felt. Conversation stays on Lightning for speed. Ultra
wasn't better here (its misses were run-to-run variance on 8 cases), so Super is the default. The one remaining
miss is a sign error in integration by parts.

**5. End to end on real handwriting** (screenshot → Super-VL → checker):

```bash
uv run experiments/exp_pipeline.py
```

Results on 5 handwritten iPad pages (2026-10-17):

| Page | Tests | Result |
|---|---|---|
| Clean algebra | no false alarm | quiet ✅ |
| Incline, cos instead of sin | conceptual error | flagged line 3 ✅ |
| Calculus with scribbled-out line | ignore crossed-out work | marked `crossed_out`, quiet ✅ |
| Lecture text notes | don't check prose | quiet ✅ |
| x² = 9 → x = 3, circled "?" | trigger + lost root | circled ? found at that line, flagged ✅ |

Reading: ~1.7 s and ~600–900 input tokens per page. Reading + checking: 2.3–6.6 s (the slow end is when Super confirms a mistake). Note: the character error rate overstates misreads — e.g. it writes `F_parallel`/`theta` for `F∥`/`θ`, which is a correct reading.

**6. Voice loop** (synthesized speech, no mic needed):

```bash
uv run experiments/exp_voice.py
```

Median over 5 questions (2026-10-17): end-of-speech wait 0.72 s + Parakeet 0.13 s (on-device) + Nemotron 3.5
Lightning first sentence 0.50 s = **1.36 s** to the first spoken sentence (plus `say` start-up). Non-wake speech
("I need to buy groceries") is ignored locally. Follow-ups ("wait, why?") use conversation memory.

## Tests

```bash
uv run pytest
```
