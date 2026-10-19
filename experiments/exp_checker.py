"""Day-1 test: can the checker cascade catch planted mistakes without false alarms?

    uv run experiments/exp_checker.py --offline   # SymPy layer only, no API key needed
    uv run experiments/exp_checker.py             # full cascade: SymPy -> Nemotron Nano -> Nemotron Super

A problem counts as "caught" if the first flagged line equals the planted error line.
"""

import argparse
import json
import statistics
from pathlib import Path

from margin.checker.cascade import WorkLine, check, report_json

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="SymPy only, no model calls")
    ap.add_argument("--data", default=str(ROOT / "data" / "planted_errors.json"))
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--screen", default=None, help="override the screening model")
    ap.add_argument("--smart", default=None, help="override the confirming model (e.g. nvidia/Nemotron-3-Ultra-550b-a55b)")
    args = ap.parse_args()

    client = None
    if not args.offline:
        from margin.nebius import Nebius

        import dataclasses

        client = Nebius()
        overrides = {k: v for k, v in {"model_screen": args.screen, "model_smart": args.smart}.items() if v}
        client.cfg = dataclasses.replace(client.cfg, **overrides)
        print(f"screen: {client.cfg.model_screen}   confirm: {client.cfg.model_smart}\n")

    problems = json.loads(Path(args.data).read_text())
    caught = missed = wrong_line = false_alarms = clean_ok = 0
    latencies = []

    for p in problems:
        work = [WorkLine(w["text"], w.get("sympy")) for w in p["work"]]
        report = check(client, p["question"], work, course=p["course"])
        first = report.flags[0].line if report.flags else None
        truth = p["error_line"]
        if truth is None:
            outcome = "ok" if first is None else "FALSE ALARM"
            clean_ok += first is None
            false_alarms += first is not None
        elif first == truth:
            outcome, caught = "caught", caught + 1
        elif first is None:
            outcome, missed = "missed", missed + 1
        else:
            outcome, wrong_line = f"wrong line ({first})", wrong_line + 1
        total = sum(report.timings.values())
        latencies.append(total)
        print(f"{p['id']:<24} truth={str(truth):<5} flagged={str(first):<5} {outcome:<16} {total:6.2f}s  calls={report.calls}")
        if args.verbose or outcome not in ("ok", "caught"):
            for f in report.flags:
                print(f"    line {f.line} ({f.confidence:.2f}): {f.reason}" + (f"\n      hint: {f.hint}" if f.hint else ""))
        if args.verbose:
            print("    " + report_json(report).replace("\n", "\n    "))

    with_errors = sum(1 for p in problems if p["error_line"] is not None)
    clean = len(problems) - with_errors
    print("\n--- summary " + ("(SymPy only)" if args.offline else "(full cascade)") + " ---")
    print(f"errors caught at the right line: {caught}/{with_errors}   missed: {missed}   wrong line: {wrong_line}")
    print(f"false alarms on correct work:    {false_alarms}/{clean}")
    print(f"latency per problem: median {statistics.median(latencies):.2f}s   max {max(latencies):.2f}s")


if __name__ == "__main__":
    main()
