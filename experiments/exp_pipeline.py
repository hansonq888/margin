"""End-to-end test on real handwriting: page screenshot -> Nemotron Super-VL reading -> checker cascade.

    uv run experiments/exp_pipeline.py

Expected mistakes are listed below for the scripted test pages (line numbers count non-crossed-out lines).
"""

import time
from pathlib import Path

from PIL import Image

from margin.checker.cascade import WorkLine, check
from margin.ink import crop_to_ink
from margin.nebius import Nebius
from margin.transcribe import transcribe

ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "Margin Test-1.jpg": None,
    "Margin Test-2.jpg": 3,  # cos instead of sin
    "Margin Test-3.jpg": None,  # the wrong line is crossed out
    "Margin Test-4.jpg": 4,  # 4 x 18 = 62
    "Margin Test-5.jpg": None,
    "Margin Test-6.jpg": 3,  # forgot x = -3
}


def main() -> None:
    client = Nebius()
    for path in sorted((ROOT / "notes").glob("*.jpg")):
        start = time.perf_counter()
        t = transcribe(client, crop_to_ink(Image.open(path)))
        lines = [l for l in t.lines if l.kind != "crossed_out"]
        work = [WorkLine(l.text, l.sympy) for l in lines]
        question = lines[0].text if lines else ""
        report = check(client, question, work)
        total = time.perf_counter() - start

        first = report.flags[0].line if report.flags else None
        expected = EXPECTED.get(path.name, "?")
        verdict = "OK" if first == expected else "MISMATCH"
        print(f"\n=== {path.name}: expected {expected}, flagged {first} -> {verdict}   "
              f"(read {t.raw.latency_s:.1f}s + check {sum(report.timings.values()):.1f}s = {total:.1f}s)")
        if t.circled_question:
            print(f"  circled ? near line {t.circled_near_line}")
        for n, w in enumerate(work, 1):
            alg = report.algebra.get(n)
            print(f"  {n}. {w.text:<40} {'[' + alg.verdict.value + ']' if alg else ''}")
        for f in report.flags:
            print(f"  >> line {f.line} ({f.confidence:.2f}): {f.reason}\n     hint: {f.hint}")


if __name__ == "__main__":
    main()
