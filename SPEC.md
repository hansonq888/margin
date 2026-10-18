# Margin — Spec

> *Working name.* A study companion that reads your handwriting as you write, knows the problem on your laptop and the course it's from, and speaks up the moment your reasoning slips — without you ever stopping to ask.

- **Hackathon:** Nebius x NVIDIA Global AI Hackathon (Devpost) — submission deadline **Oct 30, 2026**
- **Track:** Personal AI (background, private assistant with persistent memory and tool access)
- **Platforms (v1):** macOS app + iPad over USB. Backend is device-agnostic.

---

## 1. Problem

Students who take notes and do problem sets by hand on an iPad have no AI help that fits how they work:

- **Chatbots (ChatGPT, Claude):** you stop, screenshot, switch apps, explain context, and usually get the answer instead of help.
- **Screen-sharing voice assistants (Gemini Live, ChatGPT voice):** see your screen, but are *reactive* (wait for you to ask), *generic* (no idea what course or problem), *forgetful* (no memory), and stream video to a large model.
- **Note-app AI (Goodnotes Math Assist, Apple Math Notes):** Goodnotes checks handwritten math, but only inside its own marketplace practice areas; Math Notes evaluates expressions, it doesn't check reasoning.

Nothing combines **the question on your laptop + your handwritten work on the iPad + your course + your history**, and nothing catches a mistake *before you ask*.

**Why it matters:** VanLehn's 2011 meta-analysis found step-based tutoring (feedback on each step) reached an effect size of ~0.76 vs. ~0.79 for human tutors, while answer-based systems landed around 0.3–0.4. Chatbots are answer-based. Margin is step-based, living in the note app you already use. (Caveat: classroom effects are smaller — Steenbergen-Hu & Cooper, 2013.)

---

## 2. The core insight

**Efficiency is what makes it proactive.**

Margin never streams video to a big model. It compiles your study session into a small, live, structured **session state** (a few KB of JSON): the question as text, your handwritten lines as LaTeX/SymPy, your course, the recent conversation, your mistake history. Pixels are only processed once, incrementally, as small crops of new ink.

Because the state is tiny and always ready:
1. **Checking is cheap enough to run continuously** → it can speak up before you ask.
2. **Answers are fast** → no capture/OCR happens at question time.
3. **Every interaction is grounded** in the exact problem, line, and course.

> **Pitch line:** *"Gemini Live watches your screen. Margin understands your study session — and catches your mistake before you ask."*

---

## 3. User experience

### Zero-friction flow
1. **Once:** install, grant Screen Recording / Accessibility / Mic / Camera (for iPad capture) in a single onboarding screen.
2. **Every session:** plug in iPad → soft chime → "Watching: PHYS 170." Start writing. That's it.

### Ways to ask — "however your hands already are"
| Trigger | Notes |
|---|---|
| **Circled "?" in the margin** | Signature move. Also *localizes* — Margin knows which line you mean. Circled (not bare) to avoid real question marks. |
| **"Hey Margin…"** | Local wake word; nothing leaves the Mac until it fires. Toggle in menu bar. |
| **Typed `??`** | On Mac keyboard or iPad keyboard. |
| **Push-to-talk hotkey** | When at the laptop. |
| **Proactive** | Margin notices a mistake or a stall and offers help. |

### Responses
- **Voice in AirPods** (default) — 1–2 sentences, interruptible (start talking → it stops).
- **Silent mode** (library) — text on the Mac overlay; triggers via circled "?".
- **Overlay** — thumbnail of your page with the flagged line highlighted + **"here's what I read"** transcription.

### Help ladder
Nudge → pointed hint → worked step → answer. Say "more" (or circle "?" again) to climb. Default starts at nudge.

### Modes
- **Interruptiveness:** Quiet / Gentle (default) / Active.
- **Integrity:** **Practice** (full help) / **Graded** (concept explanations only, no step checking or answers). Per-course setting.

---

## 4. Architecture

```
┌────────── iPad (USB) ──────────┐   ┌──────────── Mac ─────────────┐
│ any note app (GoodNotes, etc.) │   │ Canvas / PDF / Gradescope     │
└──────────────┬─────────────────┘   └──────────────┬───────────────┘
   CoreMediaIO opt-in + AVCapture        Accessibility API / browser ext
               │ frames                              │ question text
               ▼                                     ▼
     ┌──────────────────┐                  ┌──────────────────┐
     │  Ink tracker     │ local, no AI     │ Question reader  │
     │  diff · scroll · │                  │ (+ Omni fallback │
     │  idle · crop     │                  │  for PDFs/images)│
     └────────┬─────────┘                  └────────┬─────────┘
              │ small crops of new ink              │
              ▼                                     │
     ┌──────────────────┐                           │
     │ Nemotron Nano    │ ink → LaTeX/SymPy lines   │
     │ Omni (vision)    │ + circled-? detection     │
     └────────┬─────────┘                           │
              ▼                                     ▼
     ┌──────────────────────────────────────────────────────┐
     │          SESSION STATE (few KB JSON)                  │
     │ course · question · lines · flags · talk · memory     │
     └───────┬───────────────────────────────┬───────────────┘
             │ continuously                  │ on trigger
             ▼                               ▼
   ┌───────────────────────┐      ┌───────────────────────────┐
   │ Checker cascade       │      │ Voice loop                │
   │ 1. SymPy (exact)      │      │ wake word (local)         │
   │ 2. Nemotron Nano      │      │ → Parakeet ASR (on-device)│
   │    screens (recall)   │      │ → Nemotron Nano / Super   │
   │ 3. Nemotron Super     │      │ → Magpie TTS (streaming)  │
   │    confirms (precision│      │ → AirPods                 │
   │ 4. interruption budget│      └───────────────────────────┘
   └───────────────────────┘
             │
             ▼
   Course memory: materials (embeddings) · mistake patterns · searchable notes
```

### 4.1 Capture
- **iPad:** macOS opt-in via `kCMIOHardwarePropertyAllowScreenCaptureDevices`; the plugged-in iPad appears as an `AVCaptureDevice` (muxed). No iPad app, no Wi-Fi. AirPlay mirroring is the wireless fallback (same pipeline, window capture).
- **Mac question:** Accessibility API (`AXUIElement`) for focused window text; browser extension for Canvas/Gradescope; Omni screenshot fallback for PDFs/images.

### 4.2 Ink tracker (local, no AI)
- Downscaled grayscale frame diff against a committed baseline.
- **Scroll/pan detection** via phase correlation — a page that *moved* is not new ink.
- **Page change / zoom** → large unexplained change → re-baseline.
- **Idle gate** — commit a region only after the pen has been still ~1.5 s.
- **Ignore regions** — toolbars, popovers.
- **Added vs. erased** — darker = ink added, lighter = erased → update/remove lines.
- **Notification banners** — drop frames with a banner (privacy).

### 4.3 Ink → text (Nemotron Nano Omni, reasoning off)
- Input: small crop of new ink. Output per line: `text`, `latex`, `sympy` (if pure algebra), `kind` (math/text/diagram/crossed_out), `complete`, `confidence`; plus circled-"?" detection.
- **Why transcribe-then-check:** FERMAT (ACL 2025) found several VLMs reason better about math when handwriting is replaced with printed text. So we read handwriting once, then reason over clean text.

### 4.4 Session state
```json
{
  "course": "PHYS 170",
  "question": {"id": "3b", "text": "A 2 kg block slides down a 30° incline...", "source": "Canvas"},
  "work": [
    {"line": 1, "latex": "F = mg\\sin\\theta", "sympy": "F = m*g*sin(theta)", "conf": 0.94},
    {"line": 2, "latex": "F = 2(9.8)\\cos 30^\\circ", "sympy": "F = 2*9.8*cos(30*pi/180)", "conf": 0.88}
  ],
  "flags": [{"line": 2, "type": "inconsistent_substitution", "conf": 0.86, "confirmed": true}],
  "talk": ["user: why is it sine here?"],
  "memory": ["swaps sin/cos on inclines (3x this week)"]
}
```

### 4.5 Checker cascade
1. **SymPy (exact):** for consecutive lines, test algebraic equivalence (proportional residuals, or equal solution sets for single-variable equations, with rounding tolerance). Equivalent → no AI call, no false alarm.
2. **Nemotron Nano (screen):** whole-solution pass, per-line status (ok / error / unsure / incomplete). Tuned for recall.
3. **Nemotron Super (confirm):** only for candidates (SymPy "not equivalent" or Nano "error"), with SymPy evidence. Tuned for precision. Nothing is spoken unless confirmed.
4. **Interruption budget:** high confidence, after a pause, rate-limited; mode-dependent.

Grounding: ProcessBench (ACL 2025) shows prompted critic LLMs beat most process reward models at locating the first wrong step, but small models are noticeably weaker — hence small-screens / large-confirms, with exact math wherever possible.

### 4.6 Voice loop
| Stage | Choice | Notes |
|---|---|---|
| Wake word | openWakeWord / Porcupine (local) | off by default in public |
| ASR | **NVIDIA Parakeet** (EOU 120M on Apple Neural Engine via CoreML) | streaming partials, end-of-utterance detection |
| LLM | **Nemotron Nano** (most turns, reasoning off) / **Super** (hard ones, "give me a sec") | context already compiled |
| TTS | **NVIDIA Magpie** (build.nvidia.com or self-hosted on Nebius AI Cloud) | speak from first sentence |
| Barge-in | user speech → stop audio | |

**Target:** ~1–1.5 s end-of-speech → first audio. To be measured and shown.

### 4.7 Course memory (Personal AI)
- **Materials:** drag syllabus/slides PDFs onto the menu-bar icon → chunk + embed → retrieval for hints in *your course's* notation.
- **Mistake patterns:** per course, from confirmed flags.
- **Searchable handwritten notes:** everything Margin transcribes is kept as text → "What did I write about Gauss's law last week?"
- **Session recap:** "3 problems, 2 mistakes — both sign errors. Want 2 practice problems?"
- **Course detection:** from window titles / notebook names; corrected once, remembered.

### 4.8 Model & platform map
| Role | Model | Where |
|---|---|---|
| Ink & screen reading | **Nemotron 3.5 Super-VL** (Nano Omni is not on Token Factory) | Token Factory ✅ |
| Fast screening, most replies | **Nemotron 3.5 Lightning** (or Nemotron 3 Nano 30B) | Token Factory ✅ |
| Confirmation, hard reasoning | **Nemotron 3 Super 120B** (Ultra 550B also available) | Token Factory ✅ |
| ASR / TTS | Parakeet / Magpie | on-device / NVIDIA API or Nebius AI Cloud |
| Exact algebra | SymPy | local |
| Web lookup (concepts not in materials) | Tavily | only if a Tavily prize applies |

---

## 5. Privacy & integrity
- Frames are processed locally; only small crops of new ink are sent. Notification frames are dropped. No video is stored — only text.
- Menu-bar indicator while watching; "Margin, pause"; auto-pause when non-study apps are frontmost.
- **Graded mode** per course: no step checking or answers.

---

## 6. Risks & mitigations
| Risk | Mitigation | Day-2 kill criterion |
|---|---|---|
| Handwriting misreads → false accusations | transcribe-then-check, "here's what I read", confirm before speaking, SymPy | If Omni line accuracy on real notes is poor → try Nemotron Parse / fine-tune → else pivot to teach-back partner |
| ~~Nano Omni unavailable on Token Factory~~ | confirmed unavailable — using Nemotron 3.5 Super-VL instead | resolved |
| Omni latency (7.8 s TTFT measured with reasoning at 10K tokens) | reasoning off, tiny crops; background path only | — |
| Scroll/zoom/page turns read as new ink | phase-correlation scroll detection, re-baseline on page change | — |
| Messy, non-linear scratch work | dependency-aware checking; only check `complete` lines | — |
| Annoying interruptions | interruption budget, modes, one-tap/voice dismiss that learns | — |
| Academic integrity concerns | Practice/Graded modes | — |
| Scope (13 days) | strict cut order (below) | — |

**Judge Q&A to prepare:** Gemini Live? · accuracy? · latency? · cost per study-hour? · what if wrong? · cheating? · privacy? · why Mac+iPad?

---

## 7. Evaluation (numbers to show)
- **Transcription:** line accuracy / character error rate on ~10 pages of real notes.
- **Error detection:** detection rate and false-alarm rate on planted-error set + **FERMAT** handwritten subset (`ai4bharat/FERMAT` on Hugging Face).
- **Latency:** per stage (ASR, LLM TTFT, TTS first audio) and end-to-end.
- **Efficiency:** tokens & $ per study-hour vs. streaming screenshots to a large VLM.

---

## 8. Plan (Oct 17 → Oct 30)
| Days | Goal | Gate |
|---|---|---|
| **1–2** | USB capture · Omni on real notes · SymPy+Nemotron checker on planted errors · voice latency probe | **Day 2 go/no-go** on handwriting accuracy |
| 3–5 | Ink tracker (scroll/zoom/erase/notifications) · session state · Mac question reader | |
| 6–8 | Voice loop · push-to-talk · barge-in · wake word | |
| 9–10 | Circled "?" · proactive checking + budget · help ladder · "what I read" overlay · dismiss · Practice/Graded · course detection | |
| 11 | FERMAT eval · latency/cost numbers · searchable notes + session recap | |
| 12–13 | Record video (USB, at home) · README · Nebius/NVIDIA feedback · **submit a day early** | |

**Cut order if behind:** course materials RAG → typed `??` → reading-course demo → wake word.
**Never cut:** iPad capture · proactive catch · voice.

---

## 9. Demo video (2:00)
1. **0:00** Problem: laptop ↔ iPad ↔ ChatGPT context switching.
2. **0:15** Plug in → chime → "PHYS 170, problem 3b."
3. **0:30** Write a wrong line → unprompted: *"Look at line 2 — should that still be sine?"*
4. **0:55** Circle "?" by a later step → hint for that line → "wait, why?" → interrupt mid-sentence.
5. **1:10** Say "that's fine" to a flag → it backs off gracefully.
6. **1:20** Reading course: "Hey Margin, what's this question actually asking?" (uses course slides).
7. **1:35** Engineering: live session state + real latency, cost, accuracy numbers.
8. **1:50** "Gemini Live watches your screen. Margin understands your study session."

---

## 9b. Remaining priorities (as of Oct 17)
1. ~~Mac question reader~~ — done (Oct 17): dropped PDF read locally (A) + "look at my laptop" one-shot capture (B);
   always-on watching deliberately left out for privacy
2. ~~On-screen overlay~~ — done (Oct 17): local web overlay + Swift floating panel; Socratic hint on screen, reason on demand
3. NVIDIA Magpie TTS (needs NVIDIA API key) instead of macOS `say`
4. Course memory + searchable handwritten notes
5. Numbers: FERMAT accuracy, latency, cost per study-hour
6. Demo video, README, Nebius/NVIDIA feedback (last 2 days)

**Stretch:** acoustic echo cancellation via a Swift helper using Apple Voice Processing I/O (play TTS through the
same audio engine that records the mic) so barge-in works on laptop speakers; today it relies on text-based echo
filtering and works best with AirPods.

## 10. Out of scope (v1)
Lecture mode · Windows/Android · iPad companion app · accounts/sharing · Canvas/Gradescope API integration · checking diagrams.

## 11. Submission checklist
- [ ] Working demo URL (test build) · [ ] YouTube video · [ ] public repo, OSS license, setup instructions
- [ ] Feedback on Nebius Token Factory / AI Cloud / NVIDIA models · [ ] track selected: Personal AI
