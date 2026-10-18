"""Exact (non-AI) step checking with SymPy.

Given two consecutive lines in SymPy syntax, decide whether the second follows from the first by
algebra alone. Verdicts:

  EQUIVALENT      - same equation up to a constant factor, or same solution set (with rounding slack).
                    Safe to skip AI checking for this step.
  NOT_EQUIVALENT  - same variables, but the step changes the solutions. Strong evidence of an error,
                    though some valid moves (squaring, differentiating) also land here, so an LLM confirms.
  UNKNOWN         - can't parse, different variables (substitution / new relation), or too hard to decide.
"""

import random
from dataclasses import dataclass
from enum import Enum

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCALS = {"e": sp.E, "pi": sp.pi, "ln": sp.log}
_SAMPLES = 6


class Verdict(str, Enum):
    EQUIVALENT = "equivalent"
    NOT_EQUIVALENT = "not_equivalent"
    UNKNOWN = "unknown"


@dataclass
class StepCheck:
    verdict: Verdict
    reason: str


def parse_line(src: str | None) -> list[sp.Expr] | None:
    """Parse `a = b = c` into residuals [a-b, b-c]; a bare expression becomes [expr] (no relation)."""
    if not src or not src.strip():
        return None
    parts = [p for p in src.replace("==", "=").split("=")]
    try:
        exprs = [parse_expr(p.strip(), local_dict=_LOCALS, transformations=_TRANSFORMS) for p in parts]
    except Exception:
        return None
    if any(not isinstance(e, sp.Expr) for e in exprs):
        return None
    if len(exprs) == 1:
        return None  # an expression with no "=" doesn't claim anything checkable on its own
    return [exprs[i] - exprs[i + 1] for i in range(len(exprs) - 1)]


def check_step(prev: str | None, curr: str | None, rel_tol: float = 5e-3) -> StepCheck:
    a, b = parse_line(prev), parse_line(curr)
    if a is None or b is None:
        return StepCheck(Verdict.UNKNOWN, "could not parse one of the lines")
    # Within a chained line (a = b = c) every link must hold; compare the last link of prev
    # to the first link of curr, which is how students usually continue.
    ra, rb = a[-1], b[0]
    syms_a, syms_b = ra.free_symbols, rb.free_symbols
    if syms_a != syms_b:
        return StepCheck(Verdict.UNKNOWN, "lines use different variables (substitution or new relation)")
    if not syms_a:
        return _check_numeric(ra, rb, rel_tol)
    if _proportional(ra, rb):
        return StepCheck(Verdict.EQUIVALENT, "same equation up to a constant factor")
    if len(syms_a) == 1:
        return _compare_solutions(ra, rb, next(iter(syms_a)), rel_tol)
    return StepCheck(Verdict.UNKNOWN, "multi-variable step that is not a simple rescaling")


def _check_numeric(ra: sp.Expr, rb: sp.Expr, rel_tol: float) -> StepCheck:
    """No variables: both lines are numeric claims like `2*9.8*sin(pi/6) = 9.8`."""
    try:
        va, vb = complex(sp.N(ra)), complex(sp.N(rb))
    except (TypeError, ValueError):
        return StepCheck(Verdict.UNKNOWN, "could not evaluate numerically")
    if abs(vb) > 1e-9:
        return StepCheck(Verdict.NOT_EQUIVALENT, "this numeric statement does not hold")
    if abs(va) > 1e-9:
        return StepCheck(Verdict.UNKNOWN, "the previous line was already false")
    return StepCheck(Verdict.EQUIVALENT, "both numeric statements hold")


def _proportional(ra: sp.Expr, rb: sp.Expr) -> bool:
    """True if rb == k * ra for a nonzero constant k, tested at random points (robust to floats)."""
    syms = sorted(ra.free_symbols | rb.free_symbols, key=str)
    fa, fb = sp.lambdify(syms, ra, "math"), sp.lambdify(syms, rb, "math")
    rng = random.Random(0)
    ratio = None
    hits = 0
    for _ in range(_SAMPLES * 3):
        point = [rng.uniform(0.3, 3.0) for _ in syms]
        try:
            va, vb = fa(*point), fb(*point)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue
        if isinstance(va, complex) or isinstance(vb, complex) or abs(va) < 1e-9:
            continue
        r = vb / va
        if ratio is None:
            if abs(r) < 1e-12:
                return False
            ratio = r
        elif abs(r - ratio) > 1e-6 * max(1.0, abs(ratio)):
            return False
        hits += 1
        if hits >= _SAMPLES:
            return True
    return False


def _compare_solutions(ra: sp.Expr, rb: sp.Expr, x: sp.Symbol, rel_tol: float) -> StepCheck:
    sa, sb = _real_solutions(ra, x), _real_solutions(rb, x)
    if sa is None or sb is None:
        return StepCheck(Verdict.UNKNOWN, f"could not solve for {x}")
    if _same_set(sa, sb, rel_tol):
        return StepCheck(Verdict.EQUIVALENT, f"same solutions for {x}: {_fmt(sa)}")
    return StepCheck(Verdict.NOT_EQUIVALENT, f"solutions for {x} changed from {_fmt(sa)} to {_fmt(sb)}")


def _real_solutions(expr: sp.Expr, x: sp.Symbol) -> list[float] | None:
    try:
        sols = sp.solve(sp.nsimplify(expr, rational=True), x)
    except Exception:
        return None
    out = []
    for s in sols:
        try:
            v = complex(sp.N(s))
        except (TypeError, ValueError):
            return None
        if abs(v.imag) < 1e-9:
            out.append(v.real)
    return sorted(out)


def _same_set(a: list[float], b: list[float], rel_tol: float) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(p - q) <= rel_tol * max(1.0, abs(p), abs(q)) for p, q in zip(a, b))


def _fmt(vals: list[float]) -> str:
    return "{" + ", ".join(f"{v:.4g}" for v in vals) + "}" if vals else "{}"
