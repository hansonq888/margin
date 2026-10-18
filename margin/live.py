"""Margin, live: iPad frames -> ink tracker -> Nemotron reading -> session state -> coach -> alerts.

With the capture tool running in another terminal:
    capture/.build/release/ipad-capture --out frames --fps 4
    uv run python -m margin.live --voice

Talk to Margin: say "Hey Margin, ..." / tap Enter and speak / hold Right Option (--hold-key alt_r) and speak.
Give it the assignment: --problems ps4.pdf, or drag the PDF onto this terminal and press Enter (read on the Mac).
Or say "hey Margin, look at my laptop" (or type `look`): one screenshot of the browser/PDF window, only then.
Talk about a saved page without the iPad:
    uv run python -m margin.live --voice --talk-only frames/session.json

Replay a recording instead:
    uv run python -m margin.live --replay frames/rec

The current session state is written to frames/session.json after every change.
"""

import argparse
import queue
import subprocess
import re
import sys
import threading
import time
from pathlib import Path

from PIL import Image

import json

from margin.coach import Alert, Coach
from margin.assignment import load_pdf
from margin.converse import Conversation, is_generic_help, is_goodbye, page_context, readable, speakable, wants_look
from margin.laptop import Laptop, QuestionMatcher, look_at_laptop
from margin.overlay import Overlay, page_snapshot, referenced_line
from margin.ink import InkRegion, InkTracker, PageChange, Scroll
from margin.nebius import Nebius
from margin.session import Block, Session, norm
from margin.transcribe import transcribe
from margin.voice import Question, Speaker, chime

FRAME_TIME = re.compile(r"frame_\d+_([\d.]+)\.jpg$")


def live_frames(frames_dir: Path, poll: float = 0.1):
    latest = frames_dir / "latest.jpg"
    last_mtime = 0.0
    while True:
        try:
            mtime = latest.stat().st_mtime
        except FileNotFoundError:
            yield None, time.monotonic()
            time.sleep(poll)
            continue
        if mtime == last_mtime:
            yield None, time.monotonic()  # no new frame, but let the coach deliver results
            time.sleep(poll)
            continue
        last_mtime = mtime
        try:
            img = Image.open(latest)
            img.load()
        except OSError:
            continue  # caught mid-write
        yield img, time.monotonic()


def replay_frames(rec_dir: Path, realtime: bool = False):
    """Recorded frames with their capture times. With realtime, wait between frames as they were recorded,
    so background checks race the writing exactly as they did live."""
    start_wall = start_rec = None
    for path in sorted(rec_dir.glob("frame_*.jpg")):
        m = FRAME_TIME.search(path.name)
        t = float(m.group(1)) if m else 0.0
        if realtime:
            if start_wall is None:
                start_wall, start_rec = time.monotonic(), t
            delay = (t - start_rec) - (time.monotonic() - start_wall)
            if delay > 0:
                time.sleep(delay)
            t = time.monotonic()
        yield Image.open(path), t


def idle_frames(poll: float = 0.1):
    while True:
        yield None, time.monotonic()
        time.sleep(poll)


def start_hold_key(name: str, on_down, on_up) -> None:
    """Global hold-to-talk key (needs Input Monitoring permission for your terminal)."""
    from pynput import keyboard

    key = getattr(keyboard.Key, name, None) or keyboard.KeyCode.from_char(name)
    held = {"down": False}

    def press(k):
        if k == key and not held["down"]:
            held["down"] = True
            on_down()

    def release(k):
        if k == key and held["down"]:
            held["down"] = False
            on_up()

    keyboard.Listener(on_press=press, on_release=release).start()


def describe(block: Block) -> str:
    judged = block.judgeable_problems()
    out = []
    for p, problem in enumerate(block.problems(), 1):
        cut = "" if any(problem[0] is j[0] for j in judged) else "   (cut off by the screen edge - not checked)"
        for n, line in enumerate(problem, 1):
            mark = "   <- circled ?" if line is block.circled else ""
            out.append(f"    P{p}.{n}  {line.text}{mark}{cut if n == 1 else ''}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--replay", default=None, help="directory of recorded frames to replay instead of live capture")
    ap.add_argument("--realtime", action="store_true", help="replay at the recorded speed (faithful timing)")
    ap.add_argument("--course", default="unknown")
    ap.add_argument("--say", action="store_true", help="speak alerts aloud (macOS `say`)")
    ap.add_argument("--voice", action="store_true", help="talk to Margin: \"hey Margin\", Enter, or a hold key (implies --say)")
    ap.add_argument("--hold-key", default=None, help="hold-to-talk key, e.g. alt_r (Right Option), f5, cmd_r")
    ap.add_argument("--voice-name", default=None, help="macOS voice for speech, e.g. Samantha (see `say -v ?`)")
    ap.add_argument("--debug-voice", action="store_true", help="print everything the mic hears and why")
    ap.add_argument("--follow-up", type=float, default=8.0,
                    help="seconds after Margin finishes speaking during which you can reply without \"hey Margin\" (0 = off)")
    ap.add_argument("--talk-only", default=None, help="no iPad: load this saved session JSON and just talk about it")
    ap.add_argument("--problems", default=None, help="the problem set PDF you're working on (read on this Mac)")
    ap.add_argument("--no-overlay", action="store_true", help="don't serve the on-screen overlay")
    ap.add_argument("--overlay-port", type=int, default=8765)
    ap.add_argument("--panel", action="store_true", help="show the overlay in a floating always-on-top window")
    ap.add_argument("--mode", choices=["ask", "coach"], default="ask",
                    help="ask: only speak when asked (Enter key / circled ?); coach: also speak up about mistakes unprompted")
    ap.add_argument("--quiet-seconds", type=float, default=20.0, help="coach mode: minimum gap between unprompted alerts")
    ap.add_argument("--ask-when", default=None, help="(testing) simulate an ask a few seconds after this line first appears")
    ap.add_argument("--state", default=None, help="where to write the session JSON (default: <frames>/session.json)")
    ap.add_argument("--ignore-top", type=float, default=0.09)
    ap.add_argument("--ignore-sides", type=float, default=0.04)
    ap.add_argument("--ignore-bottom", type=float, default=0.02)
    args = ap.parse_args()

    client = Nebius()
    session = Session.from_dict(json.loads(Path(args.talk_only).read_text())) if args.talk_only else Session(args.course)
    quiet = args.quiet_seconds if (args.realtime or not args.replay) else 0.0
    matcher = QuestionMatcher(client)
    coach = Coach(
        session, client, quiet_seconds=quiet, proactive=args.mode == "coach",
        question_for=lambda texts: (q.text if (q := matcher.match(texts, session.laptop)) else None),
    )
    assignments: queue.Queue = queue.Queue()  # Laptop objects to install (from drag-and-drop or "look at my laptop")
    overlay = None if args.no_overlay else Overlay(port=args.overlay_port)
    busy = {"what": None}  # "thinking" while waiting for Margin's first sentence, "looking" at the laptop
    last_frame_at = [0.0]
    watching_ipad = not (args.talk_only or args.replay)
    started_at = time.monotonic()

    def ipad_stale() -> float:
        """Seconds since the last iPad frame, if the feed has gone quiet (the iPad streams even when still)."""
        if not watching_ipad:
            return 0.0
        silent = time.monotonic() - (last_frame_at[0] or started_at)
        return silent if silent > 5 else 0.0

    feed_warned = [False]

    def matched_cached(texts):
        return matcher.cached(texts, session.laptop)

    def refresh_overlay() -> None:
        if not overlay:
            return
        a = session.laptop
        overlay.update(
            page=page_snapshot(session, coach, matched=matched_cached),
            course=None if session.course == "unknown" else session.course,
            mode=args.mode,
            assignment={"title": a.title or a.app, "count": len(a.questions)} if a else None,
            privacy={"voice": "on this Mac", "ipad": time.monotonic() - last_frame_at[0] < 3},
        )

    def set_assignment(laptop: Laptop) -> None:
        session.laptop = laptop
        if laptop.course and session.course == "unknown":
            session.course = laptop.course
        labels = ", ".join(q.label for q in laptop.questions) or "none found"
        print(f"\n📄 Assignment: {laptop.title or laptop.app}  [{laptop.app}]"
              f"{'  course ' + laptop.course if laptop.course else ''}\n   questions: {labels}")
        coach.recheck()

    def load_problems(path: str) -> None:
        try:
            assignments.put(load_pdf(path, client))
        except Exception as e:
            print(f"[assignment] couldn't read {path}: {e}")

    if args.problems:
        load_problems(args.problems)
    s, b = args.ignore_sides, args.ignore_bottom
    tracker = InkTracker(
        ignore=[(0.0, 0.0, 1.0, args.ignore_top), (0.0, 1.0 - b, 1.0, 1.0), (0.0, 0.0, s, 1.0), (1.0 - s, 0.0, 1.0, 1.0)]
    )
    if args.state:
        state_path = Path(args.state)
    elif args.replay:
        state_path = Path(args.replay).parent / "session.json"
    else:
        state_path = Path(args.frames) / "session.json"
    if args.talk_only:
        state_path = Path(args.state) if args.state else Path(args.talk_only).with_name("talk_session.json")
        frames = idle_frames()
        last = max(session.blocks.values(), key=lambda b: (b.bbox[1], b.bbox[0]), default=None)
        if last and last.problems():
            coach.focus = (last.id, last.problems()[-1])
    else:
        frames = replay_frames(Path(args.replay), args.realtime) if args.replay else live_frames(Path(args.frames))
    speaker = Speaker(voice=args.voice_name, dry_run=not (args.say or args.voice))
    conversation = Conversation(client)
    turn = [0]
    conversation_open = [True]  # False after "thanks, bye": answer, but don't open a follow-up window

    def deliver(alerts: list[Alert]) -> None:
        for a in alerts:
            tag = "🔔 MISTAKE" if a.kind == "mistake" else "💡 HELP"
            print(f"\n{tag}  block {a.block_id}, line {a.line}: {a.line_text}\n   \"{a.message}\"\n")
            if a.message:
                speaker.stop()  # newest message wins
                speaker.say(speakable(a.message))
                if overlay:
                    overlay.margin_says(a.message, done=True)
                    block = session.find_line(a.head, norm(a.line_text)) if a.head else session.blocks.get(a.block_id)
                    overlay.highlight(block.id if block else None, a.line_text, a.kind)

    def answer(q: Question) -> None:
        """Answer a spoken question: instantly if it's "help me" and we already found a mistake there,
        otherwise stream Nemotron's reply sentence by sentence into the speaker."""
        turn[0] += 1
        mine = turn[0]
        conversation_open[0] = not is_goodbye(q.text)
        speaker.stop()
        speaker.first_audio_at = None
        print(f"\n🙋 \"{q.text}\"   (via {q.via}; speech-to-text {q.asr_s * 1000:.0f} ms)")
        if overlay:
            overlay.heard(q.text)
        busy["what"] = "thinking"
        if wants_look(q.text):
            def look() -> None:
                busy["what"] = "looking"
                speaker.say("Okay, looking at your laptop.")
                laptop, message = look_at_laptop(client)
                print(f"   👀 {message}")
                if laptop and laptop.questions:
                    assignments.put(laptop)
                speaker.say(speakable(message))
                if overlay:
                    overlay.margin_says(message, done=True)
                session.talk.append({"you": q.text, "margin": message})
                busy["what"] = None

            threading.Thread(target=look, daemon=True).start()
            return
        found = coach.take_focus_mistake() if is_generic_help(q.text) else None
        page = page_context(session, coach, matched=lambda texts: matcher.cached(texts, session.laptop))
        stale = ipad_stale()
        if stale:
            page = (f"(IMPORTANT: Margin can't see the iPad right now - no new picture from it for {stale:.0f} seconds. "
                    "The page below may be out of date or empty because of that. Tell the student you can't see their "
                    "iPad at the moment and to check that it's plugged in and unlocked.)\n\n" + page)

        def run() -> None:
            parts: list[str] = []
            replies = iter([readable(found.message)]) if found else conversation.reply(q.text, page, session.course)
            for sentence in replies:
                if turn[0] != mine:
                    return  # interrupted by a newer question
                parts.append(sentence)
                speaker.say(speakable(sentence))
                busy["what"] = None
                if overlay:
                    overlay.margin_says(sentence)
                if len(parts) == 1:
                    while speaker.first_audio_at is None and turn[0] == mine:
                        time.sleep(0.01)
                    if speaker.first_audio_at:
                        source = "already found" if found else "Nemotron"
                        print(f"   ⏱ first audio {speaker.first_audio_at - q.spoke_until:.2f}s after you stopped talking ({source})")
            reply = " ".join(parts)
            busy["what"] = None
            if overlay:
                overlay.margin_says("", done=True)
                if found:
                    overlay.highlight(found.block_id, found.line_text, "mistake")
                elif coach.focus:
                    block_id, problem = coach.focus
                    n = referenced_line(reply, [l.text for l in problem])
                    if n:
                        overlay.highlight(block_id, problem[n - 1].text, "talk")
            if found:
                conversation.remember(q.text, reply)
                conversation._last_page = page
            session.talk.append({"you": q.text, "margin": reply})
            print(f"   🗣  Margin: {reply}\n")

        threading.Thread(target=run, daemon=True).start()

    voice = None
    if args.voice:
        from margin.voice import Transcriber, VoiceInput

        print("Loading Parakeet speech recognition (first run downloads ~600 MB)...")
        voice = VoiceInput(
            Transcriber(), on_wake=speaker.stop, is_speaking=lambda: speaker.speaking, spoken_text=speaker.recent_text,
            debug=args.debug_voice,
        )
        voice.start()
        if args.hold_key:
            try:
                start_hold_key(args.hold_key, lambda: (speaker.stop(), voice.hold_start()), voice.hold_end)
            except Exception as e:
                print(f"(hold-to-talk unavailable: {e}; grant Input Monitoring to your terminal)")

    asks: queue.Queue = queue.Queue()
    if not args.replay:
        # Enter alone: with --voice, "I'm about to ask" (next thing you say is the question); otherwise "help me".
        # A dragged-in PDF path + Enter loads the assignment; "look" + Enter looks at the laptop once.
        threading.Thread(target=lambda: [asks.put(line.strip()) for line in iter(sys.stdin.readline, "")], daemon=True).start()
    ask_when = norm(args.ask_when) if args.ask_when else None
    ask_at: float | None = None

    what = "talking about a saved page" if args.talk_only else "watching" + (" (replay)" if args.replay else "")
    print(f"Margin is {what} in {args.mode} mode. Session state -> {state_path}")
    if voice:
        print('Say "Hey Margin, ..." or tap Enter and speak' + (f", or hold {args.hold_key} and speak." if args.hold_key else "."))
    elif not args.replay:
        print("Press Enter any time to ask for help.")
    if not args.replay:
        print("Drag your problem-set PDF here and press Enter to give Margin the assignment; type `look` to show it your laptop.")
    if overlay:
        url = overlay.start()
        print(f"Overlay: {url}")
        if args.panel:
            panel = Path(__file__).resolve().parent.parent / "overlay-app" / ".build" / "release" / "margin-overlay"
            if panel.exists():
                subprocess.Popen([str(panel), url])
            else:
                print("(floating window not built: cd overlay-app && swift build -c release)")
        refresh_overlay()
    follow_up_due = False  # Margin spoke; open the follow-up window once it has really finished
    try:
        for frame, t in frames:
            if frame is not None:
                last_frame_at[0] = time.monotonic()
                for event in tracker.update(frame, t):
                    if isinstance(event, PageChange):
                        print(f"[page] {event.reason}")
                        if event.reason in ("start", "changing"):
                            session.clear()
                    elif isinstance(event, Scroll):
                        session.scroll(event.dx, event.dy)
                    elif isinstance(event, InkRegion):
                        reading = transcribe(client, event.crop)
                        block = session.apply(event.bbox, reading, replace=event.kind != "page", clipped=event.clipped)
                        if block is None:
                            print(f"[ink] {event.kind}: block cleared")
                        else:
                            print(f"[ink] {event.kind} -> block {block.id} v{block.version} "
                                  f"(read {reading.raw.latency_s:.1f}s)\n{describe(block)}")
                            coach.on_block(block)
                            if overlay and event.kind in ("added", "page"):
                                overlay.set_crop(block.id, event.crop)
                            if ask_when and ask_at is None and any(norm(l.text) == ask_when for l in block.work()):
                                ask_at = time.monotonic() + 8.0  # give the background check time, like a real pause
                        state_path.write_text(session.to_json())
            if ask_at is not None and time.monotonic() >= ask_at:
                asks.put("")
                ask_when, ask_at = None, None
            while not assignments.empty():
                set_assignment(assignments.get())
                state_path.write_text(session.to_json())
            while not asks.empty():
                typed = asks.get()
                path = typed.strip("'\"").replace("\\ ", " ")
                if path.lower().endswith(".pdf"):
                    threading.Thread(target=load_problems, args=(path,), daemon=True).start()
                    continue
                if typed.lower() in ("look", "l"):
                    answer(Question("look at my laptop", time.monotonic(), 0.0, "typed"))
                    continue
                if voice:
                    speaker.stop()
                    voice.arm("tap")
                    chime()
                    print("\n🎙  listening...")
                else:
                    print("\n🙋 you asked for help")
                    if overlay:
                        overlay.heard("Help me with this")
                    deliver(coach.ask())
            while voice and not voice.questions.empty():
                answer(voice.questions.get())
            # Keep the conversation going: right after Margin finishes, reply without "hey Margin".
            # (wait for 0.6 s of quiet: between streamed sentences the speaker is briefly idle)
            if speaker.speaking:
                follow_up_due = True
            elif follow_up_due and time.monotonic() - speaker.finished_at >= 0.6:
                follow_up_due = False
                if voice and args.follow_up > 0 and not voice.armed and conversation_open[0]:
                    voice.arm("follow-up", args.follow_up)
                    print(f"   👂 listening for a follow-up ({args.follow_up:.0f}s, no need to say \"hey Margin\")")
            deliver(coach.poll())
            stale = ipad_stale()
            if stale and not feed_warned[0]:
                feed_warned[0] = True
                print(f"\n⚠️  No picture from the iPad for {stale:.0f}s - Margin can't see your page. "
                      "Check the capture window, and that the iPad is plugged in and unlocked.")
            elif not stale and feed_warned[0]:
                feed_warned[0] = False
                print("\n✅ iPad picture is back.")
            if overlay:
                if stale:
                    overlay.set_status("watching", "can't see the iPad")
                elif speaker.speaking:
                    overlay.set_status("speaking")
                elif busy["what"]:
                    overlay.set_status(busy["what"])
                elif voice and voice.armed:
                    overlay.set_status("listening", "go ahead" if voice._armed_via == "follow-up" else "")
                else:
                    overlay.set_status("watching", "say “Hey Margin”" if voice else "")
                refresh_overlay()
        deliver(coach.drain())
        state_path.write_text(session.to_json())
    except KeyboardInterrupt:
        pass
    finally:
        coach.close()


if __name__ == "__main__":
    main()
