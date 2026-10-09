"""Tests for converter.py, written against API_CONTRACT.md.

Expected values are derived independently of the implementation:
- integer bits via format(n, "b")
- fraction bits via floor(frac * 2**p) (a single exact multiplication), not
  repeated doubling
- binary -> decimal via fractions.Fraction arithmetic
"""
import math
import random
from decimal import Decimal
from fractions import Fraction

import pytest

import converter
from converter import (
    DEFAULT_PRECISION,
    MAX_PRECISION,
    binary_to_decimal,
    decimal_to_binary,
    decimal_to_binary_steps,
)


# ---------------------------------------------------------------------------
# Independent oracles
# ---------------------------------------------------------------------------

def to_fraction(value):
    """Exact rational value of a test input (str parsed via Fraction, never float)."""
    if isinstance(value, str):
        return Fraction(Decimal(value.strip()))
    return Fraction(value)


def oracle_fraction_bits(frac, precision):
    """Fraction bits of 0 <= frac < 1, truncated to `precision` bits.

    Uses one exact scaled floor instead of repeated doubling.
    """
    scaled = frac * 2**precision
    n = math.floor(scaled)
    bits = format(n, "0%db" % precision)
    if scaled == n:  # terminates within precision: stop at the last 1-bit
        bits = bits.rstrip("0")
    return bits


def oracle_d2b(value, precision=DEFAULT_PRECISION):
    x = to_fraction(value)
    if x == 0:
        return "0"
    a = abs(x)
    ip = math.floor(a)
    fb = oracle_fraction_bits(a - ip, precision)
    s = ("-" if x < 0 else "") + format(ip, "b")
    if fb:
        s += "." + fb
    return s


def fraction_to_plain_decimal(x, k):
    """Exact plain-notation decimal string of non-negative x = m / 2**k."""
    scaled = x * 10**k
    assert scaled.denominator == 1
    digits = str(scaled.numerator).rjust(k + 1, "0")
    return digits[:-k] + "." + digits[-k:] if k else digits


def oracle_b2d(binary):
    """Exact Fraction value of a valid binary string."""
    s = binary.strip()
    sign = 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        s = s[1:]
    if s[:2].lower() == "0b":
        s = s[2:]
    ip, _, fp = s.partition(".")
    val = Fraction(int(ip, 2) if ip else 0)
    if fp:
        val += Fraction(int(fp, 2), 2 ** len(fp))
    return sign * val


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_default_precision_is_16():
    assert DEFAULT_PRECISION == 16


def test_max_precision_is_128():
    assert MAX_PRECISION == 128


# ---------------------------------------------------------------------------
# decimal_to_binary: integers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (1, "1"),
        (2, "10"),
        (5, "101"),
        (10, "1010"),
        (255, "11111111"),
        (256, "100000000"),
        (1023, "1111111111"),
        (-1, "-1"),
        (-10, "-1010"),
        (-255, "-11111111"),
        ("255", "11111111"),
        ("-10", "-1010"),
        ("+10", "1010"),
        ("  42  ", "101010"),
        ("007", "111"),
    ],
)
def test_integer_known_values(value, expected):
    assert decimal_to_binary(value) == expected


@pytest.mark.parametrize("n", list(range(-300, 301)))
def test_integer_sweep_matches_format_oracle(n):
    expected = format(n, "b") if n >= 0 else "-" + format(-n, "b")
    assert decimal_to_binary(n) == expected


@pytest.mark.parametrize(
    "n",
    [2**31 - 1, 2**32, 2**53 + 1, 2**63, 2**64 - 1, 10**30, 2**100, 2**100 - 1, 3**200, -(2**100)],
)
def test_large_integers_match_format_oracle(n):
    expected = format(n, "b") if n >= 0 else "-" + format(-n, "b")
    assert decimal_to_binary(n) == expected


def test_two_pow_100_is_one_followed_by_100_zeros():
    assert decimal_to_binary(2**100) == "1" + "0" * 100


def test_large_integer_given_as_string_is_exact():
    n = 123456789012345678901234567890123456789
    assert decimal_to_binary(str(n)) == format(n, "b")


def test_random_integers_match_format_oracle():
    rng = random.Random(1234)
    for _ in range(500):
        n = rng.randint(-(2**200), 2**200)
        expected = format(n, "b") if n >= 0 else "-" + format(-n, "b")
        assert decimal_to_binary(n) == expected, n


# ---------------------------------------------------------------------------
# decimal_to_binary: zero and negative zero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [0, -0, 0.0, -0.0, Decimal("0"), Decimal("-0"), Decimal("-0.000"),
     "0", "-0", "+0", "0.0", "-0.0", "-0.000", "000", ".0", "0."],
)
def test_zero_forms_give_plain_zero(value):
    assert decimal_to_binary(value) == "0"


# ---------------------------------------------------------------------------
# decimal_to_binary: fractions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("10.625", "1010.101"),
        ("0.5", "0.1"),
        ("0.75", "0.11"),
        ("-0.75", "-0.11"),
        ("0.25", "0.01"),
        ("0.125", "0.001"),
        ("0.0625", "0.0001"),
        ("3.5", "11.1"),
        ("-2.25", "-10.01"),
        (".5", "0.1"),
        ("5.", "101"),
        ("5.000", "101"),
        ("1.5000", "1.1"),
        (10.625, "1010.101"),
        (0.5, "0.1"),
        (-0.75, "-0.11"),
        (Decimal("10.625"), "1010.101"),
        (Decimal("-0.375"), "-0.011"),
    ],
)
def test_terminating_fractions_known_values(value, expected):
    assert decimal_to_binary(value) == expected


def test_point_one_precision_10_known_value():
    assert decimal_to_binary("0.1", 10) == "0.0001100110"


@pytest.mark.parametrize(
    "value, precision, expected",
    [
        # 0.1 = 0.0(0011) repeating
        ("0.1", 4, "0.0001"),
        ("0.1", 8, "0.00011001"),
        ("0.1", 16, "0.0001100110011001"),
        # 0.2 = 0.(0011) repeating
        ("0.2", 8, "0.00110011"),
        ("0.2", 16, "0.0011001100110011"),
        # 1/3 ~ 0.(01): 0.333... is slightly below 1/3 but agrees for these lengths
        ("0.3333333333", 12, "0.010101010101"),
        ("0.6", 8, "0.10011001"),
        # truncation, not rounding: 0.9999 -> all ones, never carries into "1"
        ("0.9999", 8, "0.11111111"),
        ("1.9999", 4, "1.1111"),
        # sign kept even when truncated fraction bits are all zero
        ("-0.0001", 2, "-0.00"),
        ("-0.1", 3, "-0.000"),
    ],
)
def test_repeating_fractions_truncated_exactly(value, precision, expected):
    assert decimal_to_binary(value, precision) == expected


@pytest.mark.parametrize("value", ["0.1", "0.2", "0.3", "0.7", "0.9", "-0.1", "12.3", "0.333333", "0.01"])
@pytest.mark.parametrize("precision", [1, 2, 7, 16, 33, 64, MAX_PRECISION])
def test_non_terminating_fraction_has_exactly_precision_digits(value, precision):
    result = decimal_to_binary(value, precision)
    assert len(result.split(".")[1]) == precision
    assert result == oracle_d2b(value, precision)


@pytest.mark.parametrize(
    "value",
    ["0.1", "0.2", "0.3", "1.1", "3.14159", "2.718281828", "-123.456", "0.999", "0.00001",
     "99.01", "1024.0009765625", "0.0009765625", "17.17"],
)
@pytest.mark.parametrize("precision", [1, 3, 10, DEFAULT_PRECISION, 50, MAX_PRECISION])
def test_fraction_strings_match_scaled_floor_oracle(value, precision):
    assert decimal_to_binary(value, precision) == oracle_d2b(value, precision)


def test_default_precision_used_when_omitted():
    assert decimal_to_binary("0.1") == oracle_d2b("0.1", DEFAULT_PRECISION)
    assert len(decimal_to_binary("0.1").split(".")[1]) == DEFAULT_PRECISION


def test_strings_parsed_exactly_not_via_float():
    # 0.1 as a string is exactly 1/10; the float 0.1 is a different (dyadic) number.
    s = decimal_to_binary("0.1", MAX_PRECISION)
    f = decimal_to_binary(0.1, MAX_PRECISION)
    assert s == oracle_d2b(Fraction(1, 10), MAX_PRECISION)
    assert f == oracle_d2b(Fraction(0.1), MAX_PRECISION)
    assert s != f


def test_float_input_uses_exact_binary_value_and_terminates():
    # float 0.1 = 0x3fb999999999999a is a dyadic rational that terminates in <= 55 bits
    result = decimal_to_binary(0.1, MAX_PRECISION)
    frac_bits = result.split(".")[1]
    assert result == oracle_d2b(Fraction(0.1), MAX_PRECISION)
    assert len(frac_bits) < MAX_PRECISION
    assert frac_bits.endswith("1")


def test_long_decimal_string_beyond_28_digits_is_exact():
    value = "0." + "1234567890" * 5  # 50 significant digits
    assert decimal_to_binary(value, MAX_PRECISION) == oracle_d2b(value, MAX_PRECISION)


def test_random_decimal_strings_match_oracle():
    rng = random.Random(42)
    for _ in range(300):
        ip = rng.randint(0, 10**6)
        fp = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 12)))
        sign = rng.choice(["", "-", "+"])
        value = "%s%d.%s" % (sign, ip, fp)
        p = rng.randint(1, MAX_PRECISION)
        assert decimal_to_binary(value, p) == oracle_d2b(value, p), (value, p)


def test_random_dyadic_fractions_terminate_exactly():
    rng = random.Random(7)
    for _ in range(300):
        k = rng.randint(1, 40)
        num = rng.randint(1, 2**k - 1)
        x = Fraction(num, 2**k) + rng.randint(0, 1000)
        value = fraction_to_plain_decimal(x, k)
        result = decimal_to_binary(value, 40)
        assert result == oracle_d2b(x, 40)
        if "." in result:
            assert result.endswith("1")


# ---------------------------------------------------------------------------
# decimal_to_binary: precision validation
# ---------------------------------------------------------------------------

def test_precision_one_is_allowed():
    assert decimal_to_binary("0.75", 1) == "0.1"


def test_precision_max_is_allowed():
    result = decimal_to_binary("0.1", MAX_PRECISION)
    assert len(result.split(".")[1]) == MAX_PRECISION


@pytest.mark.parametrize("precision", [0, -1, MAX_PRECISION + 1, 10**6, 1.5, "8", None])
def test_invalid_precision_raises_value_error(precision):
    with pytest.raises(ValueError):
        decimal_to_binary("0.1", precision)


@pytest.mark.parametrize("precision", [0, MAX_PRECISION + 1, 2.5])
def test_invalid_precision_raises_value_error_in_steps(precision):
    with pytest.raises(ValueError):
        decimal_to_binary_steps("0.1", precision)


# ---------------------------------------------------------------------------
# decimal_to_binary: invalid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["", "   ", ".", "-", "+", "abc", "1.2.3", "1,5", "nan", "NaN", "inf", "-inf", "Infinity",
     "0x1A", "1 0", "--1", "+-1", "1-", "0b101"],
)
def test_invalid_decimal_strings_raise_value_error(value):
    with pytest.raises(ValueError):
        decimal_to_binary(value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), Decimal("NaN"), Decimal("Infinity")])
def test_non_finite_numbers_raise_value_error(value):
    with pytest.raises(ValueError):
        decimal_to_binary(value)


@pytest.mark.parametrize("value", [True, False])
def test_bool_raises_value_error(value):
    with pytest.raises(ValueError):
        decimal_to_binary(value)


# ---------------------------------------------------------------------------
# decimal_to_binary_steps
# ---------------------------------------------------------------------------

STEP_VALUES = [
    (0, 16), (1, 16), (2, 16), (10, 16), (255, 16), (2**70 + 3, 16), (-37, 16),
    ("10.625", 16), ("0.5", 16), ("-0.75", 16), ("0.1", 10), ("0.1", MAX_PRECISION),
    ("-123.456", 20), ("0.3333333333", 30), ("-0.0001", 2), (0.1, MAX_PRECISION), ("7.0", 4),
]


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_steps_binary_matches_decimal_to_binary(value, precision):
    r = decimal_to_binary_steps(value, precision)
    assert r.binary == decimal_to_binary(value, precision)
    assert r.binary == oracle_d2b(value, precision)


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_steps_parts_and_sign_are_exact(value, precision):
    r = decimal_to_binary_steps(value, precision)
    x = to_fraction(value)
    assert r.negative == (x < 0)
    assert r.integer_part == math.floor(abs(x))
    assert r.fraction_part == abs(x) - math.floor(abs(x))
    assert isinstance(r.fraction_part, Fraction)
    assert 0 <= r.fraction_part < 1
    assert r.precision == precision


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_integer_steps_follow_division_by_two(value, precision):
    r = decimal_to_binary_steps(value, precision)
    steps = r.integer_steps
    if r.integer_part == 0:
        assert steps == ()
        return
    assert steps[0].dividend == r.integer_part
    for s in steps:
        assert s.dividend == 2 * s.quotient + s.remainder
        assert s.remainder in (0, 1)
        assert s.remainder == s.dividend % 2
    for prev, nxt in zip(steps, steps[1:]):
        assert nxt.dividend == prev.quotient
    assert steps[-1].quotient == 0


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_integer_steps_remainders_reversed_give_integer_bits(value, precision):
    r = decimal_to_binary_steps(value, precision)
    bits = "".join(str(s.remainder) for s in reversed(r.integer_steps)) or "0"
    assert bits == format(r.integer_part, "b")
    unsigned = r.binary.lstrip("-")
    assert unsigned.split(".")[0] == bits


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_fraction_steps_follow_multiplication_by_two(value, precision):
    r = decimal_to_binary_steps(value, precision)
    steps = r.fraction_steps
    assert len(steps) <= precision
    if r.fraction_part == 0:
        assert steps == ()
        return
    assert steps[0].fraction == r.fraction_part
    for s in steps:
        assert 0 < s.fraction < 1
        assert s.product == 2 * s.fraction
        assert s.bit == int(s.product)
        assert s.bit in (0, 1)
        assert s.remaining == s.product - s.bit
    for prev, nxt in zip(steps, steps[1:]):
        assert nxt.fraction == prev.remaining


@pytest.mark.parametrize("value, precision", STEP_VALUES)
def test_fraction_step_bits_equal_output_fraction_bits(value, precision):
    r = decimal_to_binary_steps(value, precision)
    bits = "".join(str(s.bit) for s in r.fraction_steps)
    unsigned = r.binary.lstrip("-")
    _, _, out_frac = unsigned.partition(".")
    assert out_frac == bits


@pytest.mark.parametrize(
    "value, precision, exact",
    [
        ("10.625", 16, True), ("0.5", 1, True), ("0.75", 1, False), ("0.75", 2, True),
        ("0.1", 10, False), ("0.1", MAX_PRECISION, False), (0.1, MAX_PRECISION, True),
        (5, 16, True), ("0", 1, True), ("-0.0001", 2, False), ("0.0625", 4, True), ("0.0625", 3, False),
    ],
)
def test_is_exact_flag(value, precision, exact):
    r = decimal_to_binary_steps(value, precision)
    assert r.is_exact is exact
    if exact:
        assert (r.fraction_steps == () and r.fraction_part == 0) or r.fraction_steps[-1].remaining == 0
    else:
        assert len(r.fraction_steps) == precision
        assert r.fraction_steps[-1].remaining != 0


def test_steps_dataclasses_are_frozen():
    r = decimal_to_binary_steps("2.5", 4)
    with pytest.raises(Exception):
        r.binary = "x"
    with pytest.raises(Exception):
        r.integer_steps[0].remainder = 5
    with pytest.raises(Exception):
        r.fraction_steps[0].bit = 5


def test_step_classes_exported():
    assert hasattr(converter, "DivisionStep")
    assert hasattr(converter, "MultiplicationStep")
    assert hasattr(converter, "ConversionResult")


def test_algorithm_terminates_quickly_for_max_precision_and_huge_integer():
    # 1/7 never terminates in binary: must stop at exactly MAX_PRECISION bits.
    value = str(2**300) + ".142857142857142857142857142857"
    r = decimal_to_binary_steps(value, MAX_PRECISION)
    assert len(r.fraction_steps) == MAX_PRECISION
    assert len(r.integer_steps) == 301
    assert r.binary == oracle_d2b(value, MAX_PRECISION)


# ---------------------------------------------------------------------------
# binary_to_decimal
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "binary, expected",
    [
        ("0", Decimal("0")),
        ("-0", Decimal("0")),
        ("1", Decimal("1")),
        ("1010", Decimal("10")),
        ("-1010", Decimal("-10")),
        ("+1010", Decimal("10")),
        ("1010.101", Decimal("10.625")),
        ("0.1", Decimal("0.5")),
        ("-0.11", Decimal("-0.75")),
        (".1", Decimal("0.5")),
        ("1.", Decimal("1")),
        ("0b1010", Decimal("10")),
        ("0B1010", Decimal("10")),
        ("-0b1.1", Decimal("-1.5")),
        ("  11111111  ", Decimal("255")),
        ("0.0001100110", Decimal("0.099609375")),
        ("0.00000001", Decimal("0.00390625")),
        ("00101", Decimal("5")),
    ],
)
def test_binary_to_decimal_known_values(binary, expected):
    result = binary_to_decimal(binary)
    assert isinstance(result, Decimal)
    assert result == expected


@pytest.mark.parametrize("n", list(range(0, 257)) + [2**64, 2**100 + 1, 3**150])
def test_binary_to_decimal_integers_match_int_oracle(n):
    assert binary_to_decimal(format(n, "b")) == Decimal(n)
    if n:
        assert binary_to_decimal("-" + format(n, "b")) == Decimal(-n)


@pytest.mark.parametrize(
    "binary",
    [
        "0." + "1" * 100,
        "1" * 80 + "." + "01" * 60,
        "-0." + "0" * 120 + "1",
        "0.0000000000000000000000000000000000000000000000000000000000000001",
        "1." + "0" * 127 + "1",
    ],
)
def test_binary_to_decimal_long_inputs_exact_beyond_28_digits(binary):
    expected = oracle_b2d(binary)
    result = binary_to_decimal(binary)
    assert Fraction(result) == expected


def test_binary_to_decimal_random_matches_fraction_oracle():
    rng = random.Random(99)
    for _ in range(300):
        ip = "".join(rng.choice("01") for _ in range(rng.randint(1, 70)))
        fp = "".join(rng.choice("01") for _ in range(rng.randint(0, 130)))
        s = rng.choice(["", "-"]) + ip + ("." + fp if fp else "")
        assert Fraction(binary_to_decimal(s)) == oracle_b2d(s), s


@pytest.mark.parametrize(
    "binary",
    ["", "   ", "2", "102", "1.0.1", "abc", ".", "0b", "-0b", "1 0", "-", "+", "0x1", "1e1", "--1", "0b0b1", "b101", "1.2"],
)
def test_binary_to_decimal_invalid_raises_value_error(binary):
    with pytest.raises(ValueError):
        binary_to_decimal(binary)


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    ["10.625", "0.5", "-0.75", "255", "0", "3.0625", "-1024.0009765625", "123456789.5",
     str(2**100), "0.00048828125"],
)
def test_round_trip_exact_when_terminating(value):
    b = decimal_to_binary(value, MAX_PRECISION)
    assert binary_to_decimal(b) == Decimal(value)


def test_round_trip_exact_for_float_inputs():
    rng = random.Random(5)
    for _ in range(200):
        x = rng.uniform(-1e6, 1e6)
        b = decimal_to_binary(x, MAX_PRECISION)
        assert Fraction(binary_to_decimal(b)) == Fraction(x), x
        assert binary_to_decimal(b) == Decimal(x)


@pytest.mark.parametrize("value", ["0.1", "0.2", "-0.3", "12.7", "-99.99", "0.333333333", "2.718281828459045"])
@pytest.mark.parametrize("precision", [1, 5, 16, 64, MAX_PRECISION])
def test_round_trip_non_terminating_is_truncated_within_bound(value, precision):
    x = to_fraction(value)
    back = Fraction(binary_to_decimal(decimal_to_binary(value, precision)))
    # truncation toward zero: magnitude never exceeds the true value
    assert abs(back) <= abs(x)
    assert abs(x) - abs(back) < Fraction(1, 2**precision)
    # sign preserved (or zero)
    assert back == 0 or (back < 0) == (x < 0)


def test_binary_to_decimal_to_binary_round_trip_random():
    rng = random.Random(2024)
    for _ in range(300):
        ip = "1" + "".join(rng.choice("01") for _ in range(rng.randint(0, 60)))
        nfrac = rng.randint(0, MAX_PRECISION)
        fp = "".join(rng.choice("01") for _ in range(nfrac)).rstrip("0")
        s = rng.choice(["", "-"]) + ip + ("." + fp if fp else "")
        d = binary_to_decimal(s)
        assert decimal_to_binary(d, MAX_PRECISION) == s, s
        assert decimal_to_binary(format(d, "f"), MAX_PRECISION) == s, s


# ---------------------------------------------------------------------------
# Round 2: input size limits, ASCII-only digits, plain-notation output
# ---------------------------------------------------------------------------

import time  # noqa: E402

from converter import MAX_DECIMAL_DIGITS, MAX_DECIMAL_EXPONENT  # noqa: E402


def test_decimal_limits_constants():
    assert MAX_DECIMAL_EXPONENT == 10_000
    assert MAX_DECIMAL_DIGITS == 10_000


@pytest.mark.parametrize(
    "value",
    [
        "1" + "0" * 20000,                      # magnitude ~10**20000
        "-1" + "0" * 20000,
        "0." + "0" * 20000 + "1",               # magnitude ~10**-20001
        "1." + "7" * 20000,                     # 20001 significant digits
        "3" * 20000,                            # 20000 significant digits, big
        Decimal("1E+20000"),
        Decimal("-1E-20000"),
        Decimal("1." + "1" * 20000),
    ],
    ids=["huge", "huge-neg", "tiny", "many-frac-digits", "many-int-digits",
         "Decimal-huge", "Decimal-tiny", "Decimal-many-digits"],
)
def test_oversized_nonzero_inputs_rejected_quickly(value):
    start = time.perf_counter()
    with pytest.raises(ValueError):
        decimal_to_binary(value, MAX_PRECISION)
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "value",
    ["0." + "0" * 50000, "-0." + "0" * 50000, "0" * 50000, "+" + "0" * 30000 + ".0",
     Decimal("0E-50000"), Decimal("-0E+50000")],
    ids=["zero-frac-50000", "neg-zero-frac-50000", "zeros-50000", "plus-zeros", "Decimal-0E-50000",
         "Decimal-neg-0E+50000"],
)
def test_long_zero_inputs_are_never_rejected(value):
    start = time.perf_counter()
    assert decimal_to_binary(value, MAX_PRECISION) == "0"
    assert time.perf_counter() - start < 2.0


@pytest.mark.parametrize(
    "value",
    ["1" + "0" * 999, "0." + "0" * 999 + "1", "9" * 5000, "12345." + "6" * 3000],
    ids=["10**999", "10**-1000", "5000 nines", "long-fraction"],
)
def test_large_inputs_within_limits_are_exact(value):
    assert decimal_to_binary(value, MAX_PRECISION) == oracle_d2b(value, MAX_PRECISION)


@pytest.mark.parametrize(
    "value",
    ["١٠", "１２", "1٢", "²", "१.5", "-٥"],
    ids=["arabic-indic-10", "fullwidth-12", "mixed", "superscript-2", "devanagari", "neg-arabic"],
)
def test_non_ascii_digits_rejected(value):
    with pytest.raises(ValueError):
        decimal_to_binary(value)


@pytest.mark.parametrize(
    "binary",
    ["0." + "0" * 100 + "1", "-0." + "0" * 40 + "1", "1" + "0" * 200, "0.000000000000000000000000000001",
     "1" + "0" * 64 + ".0001", "0", "-0"],
)
def test_binary_to_decimal_plain_format_has_no_exponent(binary):
    d = binary_to_decimal(binary)
    text = format(d, "f")
    assert "e" not in text.lower()
    assert Fraction(Decimal(text)) == oracle_b2d(binary)


def test_binary_to_decimal_two_pow_minus_30():
    d = binary_to_decimal("0.000000000000000000000000000001")
    assert format(d, "f") == "0.000000000931322574615478515625"
