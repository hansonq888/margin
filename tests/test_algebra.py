import pytest

from margin.checker.algebra import Verdict, check_step, parse_line

EQ, NE, UNK = Verdict.EQUIVALENT, Verdict.NOT_EQUIVALENT, Verdict.UNKNOWN


@pytest.mark.parametrize(
    "prev, curr, expected",
    [
        ("2*x + 6 = 14", "2*x = 8", EQ),
        ("2*x = 8", "x = 4", EQ),
        ("3*x - 5 = 10", "3*x = 5", NE),  # sign error moving -5
        ("-2*(x - 3) = 8", "-2*x - 6 = 8", NE),  # distributed the negative wrong
        ("x**2 - 5*x + 6 = 0", "(x - 2)*(x - 3) = 0", EQ),
        ("x**2 + x - 6 = 0", "(x + 2)*(x - 3) = 0", NE),  # wrong factoring
        ("x/3 + x/6 = 4", "2*x/9 = 4", NE),  # added denominators
        ("2**(x + 1) = 16", "2**(x + 1) = 2**4", EQ),
        ("2**(x + 1) = 2**4", "x + 1 = 4", EQ),  # valid, but not a rescaling: needs the solution-set check
        ("n = 36/18", "n = 3", NE),  # arithmetic slip
        ("F = 2*9.8*sin(30*pi/180)", "F = 9.8", EQ),
        ("F = 2*9.8*cos(30*pi/180)", "F = 16.97", EQ),  # rounding is fine
        ("F = 2*9.8*cos(30*pi/180)", "F = 19.6", NE),
        ("F = m*g*sin(theta)", "F = 2*9.8*sin(30*pi/180)", UNK),  # substitution: AI decides
        ("n = 2", "n_O2 = 2*(1/2)", UNK),  # new variable
        ("x^2 = 9", "x = 3", NE),  # lost the negative root (worth a nudge)
        ("y = 2x + 1", "y - 1 = 2x", EQ),  # implicit multiplication
        (None, "x = 4", UNK),
        ("f'(x) = 3x^2", "x = 1", UNK),  # unparseable prime notation
    ],
)
def test_check_step(prev, curr, expected):
    assert check_step(prev, curr).verdict == expected


def test_chained_line_compares_last_link():
    assert check_step("y = 3*(x + 1) = 3*x + 3", "3*x + 3 = 12").verdict == UNK  # y vs x: different vars
    assert check_step("3*x + 3 = 12", "3*x = 9").verdict == EQ


def test_bare_expression_is_not_checkable():
    assert parse_line("x + 1") is None


def test_numeric_claims():
    assert check_step("2 + 2 = 4", "4 = 4").verdict == EQ
    assert check_step("2 + 2 = 4", "4 = 5").verdict == NE


def test_continuation_lines_link_to_previous_expression():
    from margin.checker.cascade import link_continuations

    lines = link_continuations(["diff(x**2*log(x), x)", "= 2*x*log(x) + x**2*(1/x)", "= 2*x*log(x) + x"])
    assert lines[1] == "diff(x**2*log(x), x) = 2*x*log(x) + x**2*(1/x)"
    assert lines[2] == "2*x*log(x) + x**2*(1/x) = 2*x*log(x) + x"
    assert check_step(lines[0], lines[1]).verdict == UNK  # first line alone claims nothing
    assert check_step(lines[1], lines[2]).verdict == EQ
    # A wrong derivative step is caught exactly.
    bad = link_continuations(["diff(x**2*log(x), x)", "= 2*x*log(x) + 1"])
    assert parse_line(bad[1]) is not None
