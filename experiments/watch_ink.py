"""Live (or replayed) ink pipeline: iPad frames -> ink regions -> (optional) Omni transcription.

Live, with the capture tool running in another terminal:
    capture/.build/release/ipad-capture --out frames --fps 4
    uv run experiments/watch_ink.py --frames frames

Replay a recording (from `ipad-capture --record`):
    uv run experiments/watch_ink.py --replay frames/rec

Add --transcribe to send each region to Nemotron Omni (needs NEBIUS_API_KEY).
Crops are saved to experiments/results/crops/ so you can eyeball what would be sent.
"""

import argparse
import re
import time
from pathlib import Path

from PIL import Image

from margin.ink import InkRegion, InkTracker, PageChange, Scroll

ROOT = Path(__file__).resolve().parent
FRAME_TIME = re.compile(r"frame_\d+_([\d.]+)\.jpg$")


def live_frames(frames_dir: Path, poll: float = 0.1):
    latest = frames_dir / "latest.jpg"
    last_mtime = 0.0
    while True:
        try:
            mtime = latest.stat().st_mtime
        except FileNotFoundError:
            time.sleep(poll)
            continue
        if mtime != last_mtime:
            last_mtime = mtime
            try:
                img = Image.open(latest)
                img.load()
            except OSError:
                continue  # caught mid-write
            yield img, time.monotonic()
        else:
            time.sleep(poll)


def replay_frames(rec_dir: Path):
    for path in sorted(rec_dir.glob("frame_*.jpg")):
        m = FRAME_TIME.search(path.name)
        yield Image.open(path), float(m.group(1)) if m else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="frames", help="directory the capture tool writes latest.jpg into")
    ap.add_argument("--replay", default=None, help="directory of recorded frames to replay instead")
    ap.add_argument("--transcribe", action="store_true")
    ap.add_argument("--ignore-top", type=float, default=0.09, help="fraction of screen height to ignore (app toolbar)")
    ap.add_argument("--ignore-sides", type=float, default=0.04, help="fraction of width to ignore at each side (gutters)")
    ap.add_argument("--ignore-bottom", type=float, default=0.02, help="fraction of height to ignore at the bottom")
    args = ap.parse_args()

    client = None
    if args.transcribe:
        from margin.nebius import Nebius
        from margin.transcribe import transcribe

        client = Nebius()

    crops_dir = ROOT / "results" / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    sides, bottom = args.ignore_sides, args.ignore_bottom
    tracker = InkTracker(
        ignore=[
            (0.0, 0.0, 1.0, args.ignore_top),
            (0.0, 1.0 - bottom, 1.0, 1.0),
            (0.0, 0.0, sides, 1.0),
            (1.0 - sides, 0.0, 1.0, 1.0),
        ]
    )
    frames = replay_frames(Path(args.replay)) if args.replay else live_frames(Path(args.frames))
    n = 0

    print("Watching for ink... (Ctrl-C to stop)")
    for frame, t in frames:
        for event in tracker.update(frame, t):
            if isinstance(event, Scroll):
                print(f"[scroll] dx={event.dx} dy={event.dy}")
            elif isinstance(event, PageChange):
                print(f"[page] {event.reason}")
            elif isinstance(event, InkRegion):
                n += 1
                path = crops_dir / f"region_{n:04d}_{event.kind}.png"
                event.crop.save(path)
                print(f"[ink] {event.kind} bbox={event.bbox} px={event.changed_px} -> {path.name}")
                if client and event.kind in ("added", "page"):
                    result = transcribe(client, event.crop)
                    print(f"      Omni {result.raw.latency_s:.2f}s{'  ERROR ' + result.raw.error if result.raw.error else ''}")
                    for line in result.lines:
                        print(f"      [{line.kind} {line.confidence:.2f}] {line.text}" + (f"   sympy: {line.sympy}" if line.sympy else ""))
                    if result.circled_question:
                        print(f"      >>> circled ? near line {result.circled_near_line}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
