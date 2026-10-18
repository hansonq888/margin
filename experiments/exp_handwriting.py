"""Day-1 test: how well does Nemotron Omni read YOUR handwriting, and how fast?

Put screenshots/photos of your notes in experiments/notes/ (png/jpg). Optionally add a ground-truth
file next to each image with the same name and a .txt extension (one written line per line).

    uv run experiments/exp_handwriting.py [--dir experiments/notes] [--model MODEL_ID]

Writes experiments/results/handwriting.jsonl and prints a summary.
"""

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from margin.ink import crop_to_ink
from margin.nebius import Nebius
from margin.transcribe import transcribe

ROOT = Path(__file__).resolve().parent


def char_error_rate(truth: str, guess: str) -> float:
    a, b = _normalize(truth), _normalize(guess)
    if not a:
        return 0.0 if not b else 1.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / len(a)


def _normalize(s: str) -> str:
    return "".join(s.lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "notes"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--full-page", action="store_true", help="send the whole page instead of cropping to the ink")
    args = ap.parse_args()

    images = sorted(p for p in Path(args.dir).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".heic"})
    if not images:
        raise SystemExit(f"No images in {args.dir}. Add screenshots of your handwritten notes.")

    client = Nebius()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "handwriting.jsonl"
    latencies, cers, failures = [], [], 0

    with out_path.open("w") as out:
        for path in images:
            img = Image.open(path)
            if not args.full_page:
                img = crop_to_ink(img)
            t = transcribe(client, img, model=args.model)
            row = {
                "image": path.name,
                "sent_size": list(img.size),
                "latency_s": round(t.raw.latency_s, 3),
                "usage": t.raw.usage,
                "error": t.raw.error,
                "circled_question": t.circled_question,
                "lines": [asdict(line) for line in t.lines],
            }
            truth_path = path.with_suffix(".txt")
            if truth_path.exists():
                guess = "\n".join(line.text for line in t.lines)
                row["cer"] = round(char_error_rate(truth_path.read_text(), guess), 3)
                cers.append(row["cer"])
            if t.raw.error:
                failures += 1
            latencies.append(t.raw.latency_s)
            out.write(json.dumps(row) + "\n")

            print(f"\n=== {path.name}  ({t.raw.latency_s:.2f}s{', CER ' + str(row['cer']) if 'cer' in row else ''})")
            if t.raw.error:
                print(f"  ERROR: {t.raw.error}\n  raw: {t.raw.text[:300]}")
            for line in t.lines:
                flag = "" if line.complete else "  [incomplete]"
                print(f"  [{line.kind:<11} {line.confidence:.2f}] {line.text}{flag}")
                if line.sympy:
                    print(f"  {'':>18} sympy: {line.sympy}")

    print("\n--- summary ---")
    print(f"images: {len(images)}   failures: {failures}")
    print(f"latency: median {statistics.median(latencies):.2f}s   max {max(latencies):.2f}s")
    if cers:
        print(f"character error rate: mean {statistics.mean(cers):.3f}   (on {len(cers)} images with ground truth)")
    print(f"details: {out_path}")


if __name__ == "__main__":
    main()
