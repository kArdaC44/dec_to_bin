"""Tests for ieee754.py, written against API_CONTRACT.md.

Independent oracle: struct.pack / struct.unpack on big-endian bytes, plus
hand-derived reference hex values from the contract.
"""
import math
import random
import struct

import pytest

import ieee754
from ieee754 import FORMATS, decode, encode


CODES = {"float32": "f", "float64": "d"}
WIDTH = {"float32": 32, "float64": 64}


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def oracle_bits(x, fmt):
    """Bit string of x in fmt (finite overflow in float32 -> signed infinity)."""
    try:
        raw = struct.pack(">" + CODES[fmt], x)
    except OverflowError:
        raw = struct.pack(">" + CODES[fmt], math.copysign(math.inf, x))
    return format(int.from_bytes(raw, "big"), "0%db" % WIDTH[fmt])


def oracle_value_from_bits(bits, fmt):
    raw = int(bits, 2).to_bytes(WIDTH[fmt] // 8, "big")
    return struct.unpack(">" + CODES[fmt], raw)[0]


def oracle_stored(x, fmt):
    return oracle_value_from_bits(oracle_bits(x, fmt), fmt)


def same_float(a, b):
    """Bitwise float64 equality, treating every nan as equal to every nan."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return struct.pack(">d", a) == struct.pack(">d", b)


def hex_to_bits(h, width):
    return format(int(h, 16), "0%db" % width)


# ---------------------------------------------------------------------------
# FORMATS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, total, exp, mant, bias, code",
    [("float32", 32, 8, 23, 127, "f"), ("float64", 64, 11, 52, 1023, "d")],
)
def test_format_definitions(name, total, exp, mant, bias, code):
    f = FORMATS[name]
    assert f.name == name
    assert f.total_bits == total
    assert f.exponent_bits == exp
    assert f.mantissa_bits == mant
    assert f.bias == bias == 2 ** (exp - 1) - 1
    assert f.struct_code == code
    assert 1 + f.exponent_bits + f.mantissa_bits == f.total_bits


# ---------------------------------------------------------------------------
# Reference values
# ---------------------------------------------------------------------------

FLOAT32_REFS = [
    (1.0, "0x3f800000"),
    (-2.0, "0xc0000000"),
    (10.0, "0x41200000"),
    (0.1, "0x3dcccccd"),
    (math.inf, "0x7f800000"),
    (-math.inf, "0xff800000"),
    (0.0, "0x00000000"),
    (-0.0, "0x80000000"),
    (2.0**-149, "0x00000001"),
    ((2**23 - 1) * 2.0**-149, "0x007fffff"),
    (2.0**-126, "0x00800000"),
    (3.4028234663852886e38, "0x7f7fffff"),
    (0.5, "0x3f000000"),
    (-1.5, "0xbfc00000"),
]

FLOAT64_REFS = [
    (1.0, "0x3ff0000000000000"),
    (0.1, "0x3fb999999999999a"),
    (-0.0, "0x8000000000000000"),
    (0.0, "0x0000000000000000"),
    (5e-324, "0x0000000000000001"),
    (math.inf, "0x7ff0000000000000"),
    (-math.inf, "0xfff0000000000000"),
    (1.7976931348623157e308, "0x7fefffffffffffff"),
    (-2.0, "0xc000000000000000"),
    (2.0**-1022, "0x0010000000000000"),
]


@pytest.mark.parametrize("value, hexstr", FLOAT32_REFS)
def test_float32_reference_hex(value, hexstr):
    r = encode(value, "float32")
    assert r.hex == hexstr
    assert r.bits == hex_to_bits(hexstr, 32)


@pytest.mark.parametrize("value, hexstr", FLOAT64_REFS)
def test_float64_reference_hex(value, hexstr):
    r = encode(value, "float64")
    assert r.hex == hexstr
    assert r.bits == hex_to_bits(hexstr, 64)


def test_default_format_is_float32():
    assert encode(1.0).hex == "0x3f800000"
    assert encode(1.0).fmt.name == "float32"
    assert decode("00111111100000000000000000000000") == 1.0


# ---------------------------------------------------------------------------
# Field structure
# ---------------------------------------------------------------------------

FIELD_VALUES = [0.0, -0.0, 1.0, -2.0, 0.1, 10.0, 2.0**-149, 5e-324, 1e-310, 3.4e38, 1.7e308,
                math.inf, -math.inf, math.nan, 123.456, -7.25e-20]


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", FIELD_VALUES)
def test_field_lengths_and_slicing(value, fmt):
    r = encode(value, fmt)
    f = FORMATS[fmt]
    assert r.fmt == f
    assert len(r.bits) == f.total_bits
    assert set(r.bits) <= {"0", "1"}
    assert len(r.sign_bit) == 1
    assert len(r.exponent_bits) == f.exponent_bits
    assert len(r.mantissa_bits) == f.mantissa_bits
    assert r.sign_bit + r.exponent_bits + r.mantissa_bits == r.bits
    assert r.raw_exponent == int(r.exponent_bits, 2)
    assert r.bias == f.bias
    assert r.is_negative == (r.sign_bit == "1")
    assert r.hex == "0x" + format(int(r.bits, 2), "0%dx" % (f.total_bits // 4))


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", [v for v in FIELD_VALUES if not math.isnan(v)])
def test_bits_match_struct_oracle(value, fmt):
    assert encode(value, fmt).bits == oracle_bits(value, fmt)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", FIELD_VALUES)
def test_input_value_is_float64_of_input(value, fmt):
    assert same_float(encode(value, fmt).input_value, float(value))


# ---------------------------------------------------------------------------
# Categories and exponents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, fmt, category, actual_exponent",
    [
        (1.0, "float32", "normal", 0),
        (10.0, "float32", "normal", 3),
        (0.1, "float32", "normal", -4),
        (-2.0, "float32", "normal", 1),
        (2.0**-126, "float32", "normal", -126),
        (3.4028234663852886e38, "float32", "normal", 127),
        (2.0**-149, "float32", "subnormal", -126),
        ((2**23 - 1) * 2.0**-149, "float32", "subnormal", -126),
        (0.0, "float32", "zero", None),
        (-0.0, "float32", "zero", None),
        (math.inf, "float32", "infinity", None),
        (-math.inf, "float32", "infinity", None),
        (math.nan, "float32", "nan", None),
        (1.0, "float64", "normal", 0),
        (0.1, "float64", "normal", -4),
        (1.7976931348623157e308, "float64", "normal", 1023),
        (2.0**-1022, "float64", "normal", -1022),
        (5e-324, "float64", "subnormal", -1022),
        (1e-310, "float64", "subnormal", -1022),
        (0.0, "float64", "zero", None),
        (-0.0, "float64", "zero", None),
        (math.inf, "float64", "infinity", None),
        (-math.inf, "float64", "infinity", None),
        (math.nan, "float64", "nan", None),
        # values not representable in float32 but fine in float64
        (1e-40, "float32", "subnormal", -126),
        (1e-50, "float32", "zero", None),
        (1e39, "float32", "infinity", None),
    ],
)
def test_category_and_actual_exponent(value, fmt, category, actual_exponent):
    r = encode(value, fmt)
    assert r.category == category
    assert r.actual_exponent == actual_exponent


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", [1.0, 0.1, 1e-40, 3.0e-39, 1e-310, 6.02e23, -9.5])
def test_actual_exponent_formula(value, fmt):
    r = encode(value, fmt)
    if r.category == "normal":
        assert r.actual_exponent == r.raw_exponent - r.bias
        assert 0 < r.raw_exponent < 2 ** FORMATS[fmt].exponent_bits - 1
    elif r.category == "subnormal":
        assert r.raw_exponent == 0
        assert r.actual_exponent == 1 - r.bias


# ---------------------------------------------------------------------------
# Special values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_negative_zero(fmt):
    r = encode(-0.0, fmt)
    assert r.is_negative is True
    assert r.sign_bit == "1"
    assert r.category == "zero"
    assert set(r.exponent_bits) == {"0"} and set(r.mantissa_bits) == {"0"}
    assert r.stored_value == 0.0 and math.copysign(1.0, r.stored_value) == -1.0


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_positive_zero_not_negative(fmt):
    r = encode(0.0, fmt)
    assert r.is_negative is False
    assert math.copysign(1.0, r.stored_value) == 1.0


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", [math.inf, -math.inf])
def test_infinities(value, fmt):
    r = encode(value, fmt)
    assert r.category == "infinity"
    assert set(r.exponent_bits) == {"1"}
    assert set(r.mantissa_bits) == {"0"}
    assert r.stored_value == value
    assert r.is_negative == (value < 0)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("value", [math.nan, "nan", "NaN"])
def test_nan(value, fmt):
    r = encode(value, fmt)
    assert r.category == "nan"
    assert set(r.exponent_bits) == {"1"}
    assert "1" in r.mantissa_bits
    assert math.isnan(r.stored_value)
    assert math.isnan(r.input_value)
    assert r.actual_exponent is None


# ---------------------------------------------------------------------------
# Rounding / stored value
# ---------------------------------------------------------------------------

def test_float32_point_one_is_rounded():
    r = encode(0.1, "float32")
    assert r.stored_value != 0.1
    assert r.stored_value == struct.unpack(">f", struct.pack(">f", 0.1))[0]
    assert r.stored_value == 0.10000000149011612


def test_float64_point_one_is_stored_as_is():
    assert encode(0.1, "float64").stored_value == 0.1


@pytest.mark.parametrize("value, expected_hex", [(1e39, "0x7f800000"), (-1e39, "0xff800000"),
                                                 (1e300, "0x7f800000"), (-1.7e308, "0xff800000")])
def test_float32_overflow_rounds_to_infinity(value, expected_hex):
    r = encode(value, "float32")
    assert r.hex == expected_hex
    assert r.category == "infinity"
    assert r.stored_value == math.copysign(math.inf, value)
    assert r.input_value == value


@pytest.mark.parametrize("value", [1e-50, -1e-50])
def test_float32_underflow_rounds_to_signed_zero(value):
    r = encode(value, "float32")
    assert r.category == "zero"
    assert r.stored_value == 0.0
    assert math.copysign(1.0, r.stored_value) == math.copysign(1.0, value)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_stored_value_matches_struct_round_trip_random(fmt):
    rng = random.Random(31337)
    for _ in range(500):
        x = rng.choice([rng.uniform(-1e6, 1e6), rng.uniform(-1, 1) * 10.0 ** rng.randint(-60, 60)])
        r = encode(x, fmt)
        assert same_float(r.stored_value, oracle_stored(x, fmt)), x
        assert r.bits == oracle_bits(x, fmt), x


# ---------------------------------------------------------------------------
# String inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text, fmt, expected_hex",
    [
        ("inf", "float32", "0x7f800000"),
        ("-inf", "float32", "0xff800000"),
        ("-0", "float32", "0x80000000"),
        ("-0", "float64", "0x8000000000000000"),
        ("1e-45", "float32", "0x00000001"),
        ("  1.0  ", "float32", "0x3f800000"),
        ("0.1", "float64", "0x3fb999999999999a"),
        ("10", "float32", "0x41200000"),
        ("Infinity", "float64", "0x7ff0000000000000"),
    ],
)
def test_string_inputs(text, fmt, expected_hex):
    assert encode(text, fmt).hex == expected_hex


def test_string_minus_zero_is_negative_zero():
    r = encode("-0", "float32")
    assert r.is_negative is True
    assert r.category == "zero"


def test_string_1e_minus_45_is_smallest_float32_subnormal():
    r = encode("1e-45", "float32")
    assert r.category == "subnormal"
    assert r.stored_value == 2.0**-149


def test_int_input_accepted():
    assert encode(10, "float32").hex == "0x41200000"
    assert encode(-2, "float64").hex == "0xc000000000000000"


@pytest.mark.parametrize("text", ["", "   ", "abc", "1.2.3", "0x1p3", "1,5", "--1"])
def test_encode_invalid_string_raises_value_error(text):
    with pytest.raises(ValueError):
        encode(text, "float32")


@pytest.mark.parametrize("fmt", ["float16", "float128", "", "FLOAT", "double", None])
def test_encode_invalid_format_raises_value_error(fmt):
    with pytest.raises(ValueError):
        encode(1.0, fmt)


@pytest.mark.parametrize("fmt", ["float16", "", "float"])
def test_decode_invalid_format_raises_value_error(fmt):
    with pytest.raises(ValueError):
        decode("0" * 32, fmt)


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, hexstr", FLOAT32_REFS)
def test_decode_float32_references(value, hexstr):
    out = decode(hex_to_bits(hexstr, 32), "float32")
    assert same_float(out, oracle_stored(value, "float32"))


@pytest.mark.parametrize("value, hexstr", FLOAT64_REFS)
def test_decode_float64_references(value, hexstr):
    out = decode(hex_to_bits(hexstr, 64), "float64")
    assert same_float(out, value)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_negative_zero_preserves_sign(fmt):
    bits = "1" + "0" * (WIDTH[fmt] - 1)
    out = decode(bits, fmt)
    assert out == 0.0
    assert math.copysign(1.0, out) == -1.0


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_positive_zero(fmt):
    out = decode("0" * WIDTH[fmt], fmt)
    assert out == 0.0
    assert math.copysign(1.0, out) == 1.0


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_nan(fmt):
    e = FORMATS[fmt].exponent_bits
    m = FORMATS[fmt].mantissa_bits
    assert math.isnan(decode("0" + "1" * e + "1" + "0" * (m - 1), fmt))
    assert math.isnan(decode("1" + "1" * e + "0" * (m - 1) + "1", fmt))


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_infinities(fmt):
    e = FORMATS[fmt].exponent_bits
    m = FORMATS[fmt].mantissa_bits
    assert decode("0" + "1" * e + "0" * m, fmt) == math.inf
    assert decode("1" + "1" * e + "0" * m, fmt) == -math.inf


def test_decode_ignores_spaces():
    assert decode("0 01111111 00000000000000000000000", "float32") == 1.0
    assert decode("0011 1111 1000 0000 0000 0000 0000 0000", "float32") == 1.0


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_random_bit_patterns_match_struct_oracle(fmt):
    rng = random.Random(20261009)
    width = WIDTH[fmt]
    for _ in range(3000):
        bits = format(rng.getrandbits(width), "0%db" % width)
        assert same_float(decode(bits, fmt), oracle_value_from_bits(bits, fmt)), bits


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_decode_random_subnormals_and_edges_match_struct_oracle(fmt):
    rng = random.Random(77)
    f = FORMATS[fmt]
    for _ in range(1000):
        sign = rng.choice("01")
        exp = rng.choice(["0" * f.exponent_bits, "0" * (f.exponent_bits - 1) + "1",
                          "1" * (f.exponent_bits - 1) + "0", "1" * f.exponent_bits])
        mant = format(rng.getrandbits(f.mantissa_bits), "0%db" % f.mantissa_bits)
        bits = sign + exp + mant
        assert same_float(decode(bits, fmt), oracle_value_from_bits(bits, fmt)), bits


@pytest.mark.parametrize("fmt", ["float32", "float64"])
def test_encode_random_bit_patterns_round_trip(fmt):
    rng = random.Random(4242)
    width = WIDTH[fmt]
    for _ in range(2000):
        bits = format(rng.getrandbits(width), "0%db" % width)
        value = oracle_value_from_bits(bits, fmt)
        r = encode(value, fmt)
        if math.isnan(value):
            assert r.category == "nan"
        else:
            assert r.bits == bits
            assert same_float(r.stored_value, value)


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize(
    "value",
    FIELD_VALUES + [1e39, -1e39, 1e-50, -1e-50, 0.3, -123456.789, 2.0**-140, 1e-320, "-0", "1e-45", "inf"],
)
def test_decode_of_encode_equals_stored_value(value, fmt):
    r = encode(value, fmt)
    assert same_float(decode(r.bits, fmt), r.stored_value)


@pytest.mark.parametrize(
    "bits, fmt",
    [
        ("0" * 31, "float32"),
        ("0" * 33, "float32"),
        ("0" * 64, "float32"),
        ("0" * 32, "float64"),
        ("0" * 63, "float64"),
        ("", "float32"),
        ("2" + "0" * 31, "float32"),
        ("0" * 31 + "a", "float32"),
        ("0b" + "0" * 30, "float32"),
        ("0" * 31 + "-", "float32"),
    ],
)
def test_decode_invalid_bits_raise_value_error(bits, fmt):
    with pytest.raises(ValueError):
        decode(bits, fmt)


@pytest.mark.parametrize("module_name", ["converter", "ieee754"])
def test_core_modules_do_not_import_streamlit(module_name):
    import importlib
    import pathlib
    mod = importlib.import_module(module_name)
    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "import streamlit" not in source
    assert "from streamlit" not in source


# ---------------------------------------------------------------------------
# Round 2: exact single rounding of str/int input into float32
# ---------------------------------------------------------------------------

from decimal import Decimal  # noqa: E402
from fractions import Fraction  # noqa: E402

from ieee754 import RoundingInfo, describe_rounding, exact_value  # noqa: E402

F32_INF_PATTERN = 0x7F800000


def f32_value(pattern):
    """Exact value of a non-negative float32 pattern; the inf pattern stands for 2**128.

    Treating 0x7f800000 as 2**128 makes overflow follow IEEE round-to-nearest
    with an unbounded exponent (ties at the overflow boundary go to the even
    pattern, which is infinity).
    """
    if pattern == F32_INF_PATTERN:
        return Fraction(2) ** 128
    return Fraction(struct.unpack(">f", pattern.to_bytes(4, "big"))[0])


def oracle_f32_round(q):
    """Round the exact rational q to float32 (nearest, ties-to-even); return the 32-bit pattern.

    Positive float32 patterns are ordered like their values, so binary-search
    the largest pattern <= |q| and pick the nearer neighbour.
    """
    sign = 0x80000000 if q < 0 else 0
    a = abs(q)
    lo, hi = 0, F32_INF_PATTERN
    if a >= f32_value(hi):
        return sign | F32_INF_PATTERN
    while lo < hi:  # largest p in [0, hi) with value(p) <= a
        mid = (lo + hi + 1) // 2
        if f32_value(mid) <= a:
            lo = mid
        else:
            hi = mid - 1
    p = lo
    below = a - f32_value(p)
    above = f32_value(p + 1) - a
    if above < below or (above == below and p % 2 == 1):
        p += 1
    return sign | p


def oracle_f32_hex(value):
    if isinstance(value, str):
        d = Decimal(value.strip())
        pattern = oracle_f32_round(Fraction(d))
        if d.is_zero() and d.is_signed():  # Fraction has no -0; keep the sign bit
            pattern |= 0x80000000
    else:
        pattern = oracle_f32_round(Fraction(value))
    return "0x%08x" % pattern


def dyadic_str(q):
    """Exact scientific-notation string for a dyadic rational q = n / 2**k."""
    k = q.denominator.bit_length() - 1
    assert q.denominator == 2**k
    return "%de-%d" % (q.numerator * 5**k, k)


def test_oracle_sanity_on_reference_values():
    assert oracle_f32_hex("1") == "0x3f800000"
    assert oracle_f32_hex("0.1") == "0x3dcccccd"
    assert oracle_f32_hex("1e39") == "0x7f800000"
    assert oracle_f32_hex(dyadic_str(Fraction(1, 2**149))) == "0x00000001"


@pytest.mark.parametrize(
    "text, expected_hex",
    [
        ("1.00000005960464477539062500000000001", "0x3f800001"),  # just above the 1+2**-24 tie
        ("1.000000059604644775390625", "0x3f800000"),             # exact tie 1+2**-24 -> even
        ("1.0000000596046447753906249999", "0x3f800000"),          # just below the tie
        ("1.000000178813934326171875", "0x3f800002"),             # tie 1+3*2**-24 -> even (up)
        (dyadic_str(Fraction(1, 2**150)), "0x00000000"),          # tie between 0 and 2**-149 -> 0
        (dyadic_str(Fraction(3, 2**150)), "0x00000002"),          # tie between 2**-149 and 2**-148 -> even
        (dyadic_str(Fraction(5, 2**151)), "0x00000001"),          # 1.25 * 2**-149 -> nearest 1
        ("-" + dyadic_str(Fraction(1, 2**150)), "0x80000000"),    # signed zero on underflow
        ("340282356779733661637539395458142568448", "0x7f800000"),  # overflow-boundary tie -> inf
        ("340282356779733661637539395458142568447", "0x7f7fffff"),  # just below -> FLT_MAX
        ("-340282356779733661637539395458142568448", "0xff800000"),
        ("16777217", "0x4b800000"),  # 2**24 + 1 tie -> 2**24
        ("16777219", "0x4b800002"),  # 2**24 + 3 tie -> 2**24 + 4
        ("1e-45", "0x00000001"),
        ("7e-46", "0x00000000"),     # below half of 2**-149
        ("8e-46", "0x00000001"),     # above half of 2**-149
    ],
)
def test_float32_string_rounds_exact_value_once(text, expected_hex):
    assert oracle_f32_hex(text) == expected_hex  # the oracle agrees with the hand derivation
    assert encode(text, "float32").hex == expected_hex


@pytest.mark.parametrize(
    "value, expected_hex",
    [(16777217, "0x4b800000"), (16777219, "0x4b800002"), (2**24 + 2, "0x4b800001"),
     (2**128 - 2**103, "0x7f800000"), (2**128 - 2**103 - 1, "0x7f7fffff"), (10**400, "0x7f800000"),
     (-(10**400), "0xff800000"), (2**200, "0x7f800000")],
)
def test_float32_int_rounds_exact_value_once(value, expected_hex):
    assert oracle_f32_hex(value) == expected_hex
    assert encode(value, "float32").hex == expected_hex


def test_float32_double_rounding_is_avoided():
    # 1 + 2**-24 + 2**-60 rounds up in float32, but going through float64 first
    # lands exactly on the tie 1 + 2**-24, which would then round down to 1.0.
    q = 1 + Fraction(1, 2**24) + Fraction(1, 2**60)
    text = dyadic_str(q)
    assert float(text) == 1 + 2.0**-24  # the float64 step produces an exact tie
    assert encode(text, "float32").hex == "0x3f800001"


def test_float32_random_decimal_strings_match_exact_oracle():
    rng = random.Random(8675309)
    for _ in range(1500):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 40)))
        exp = rng.randint(-60, 45)
        text = "%s%s.%se%d" % (rng.choice(["", "-"]), digits[0], digits[1:] or "0", exp)
        assert encode(text, "float32").hex == oracle_f32_hex(text), text


def test_float32_random_exact_ties_and_neighbours_match_oracle():
    rng = random.Random(1618)
    eps = Fraction(1, 2**200)
    for _ in range(800):
        p = rng.choice([rng.randrange(0, 0x00800000), rng.randrange(0, 0x7F7FFFFF),
                        rng.randrange(0x7F000000, 0x7F7FFFFF)])
        mid = (f32_value(p) + f32_value(p + 1)) / 2
        for q in (mid, mid + eps, mid - eps):
            text = dyadic_str(q)
            assert encode(text, "float32").hex == "0x%08x" % oracle_f32_round(q), (p, text[:60])
            assert encode("-" + text, "float32").hex == "0x%08x" % oracle_f32_round(-q), (p, text[:60])


def test_float32_random_ints_match_oracle():
    rng = random.Random(55)
    for _ in range(1000):
        n = rng.randint(-(2**130), 2**130) >> rng.randint(0, 128)
        assert encode(n, "float32").hex == oracle_f32_hex(n), n


@pytest.mark.parametrize("text", ["1e400", "-1e400", "1e-2000", "-1e-2000", "0.1", "16777217", "-0"])
def test_float64_string_matches_correctly_rounded_float(text):
    assert encode(text, "float64").bits == oracle_bits(float(text), "float64")


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("text, negative", [("1e-2000", False), ("-1e-2000", True)])
def test_extreme_underflow_stores_signed_zero(text, negative, fmt):
    r = encode(text, fmt)
    assert r.category == "zero"
    assert r.stored_value == 0.0
    assert (math.copysign(1.0, r.stored_value) < 0) == negative
    assert r.is_negative == negative


@pytest.mark.parametrize("fmt", ["float32", "float64"])
@pytest.mark.parametrize("text", ["1e400", "-1e400"])
def test_extreme_overflow_stores_infinity(text, fmt):
    r = encode(text, fmt)
    assert r.category == "infinity"
    assert r.stored_value == float(text)


# ---------------------------------------------------------------------------
# Round 2: exact_value / describe_rounding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("0.1", Decimal("0.1")),
        ("  -2.5 ", Decimal("-2.5")),
        ("1e400", Decimal("1e400")),
        ("16777217", Decimal(16777217)),
        (10, Decimal(10)),
        (10**30 + 1, Decimal(10**30 + 1)),
        (0.1, Decimal(0.1)),  # the float's exact binary value, not one tenth
        ("0.000000000931322574615478515625", Decimal("0.000000000931322574615478515625")),
    ],
)
def test_exact_value(value, expected):
    result = exact_value(value)
    assert isinstance(result, Decimal)
    assert result == expected


@pytest.mark.parametrize("value", ["inf", "-inf", "nan", "Infinity", math.inf, math.nan])
def test_exact_value_non_finite_is_none(value):
    assert exact_value(value) is None


@pytest.mark.parametrize("value", ["-0", "0", "-0.0", -0.0])
def test_exact_value_zero(value):
    assert exact_value(value) == 0


@pytest.mark.parametrize("value", ["", "abc", "1.2.3", "0x10"])
def test_exact_value_invalid_raises(value):
    with pytest.raises(ValueError):
        exact_value(value)


@pytest.mark.parametrize(
    "value, fmt, kind",
    [
        ("1e400", "float32", "overflow"),
        ("1e400", "float64", "overflow"),
        ("-1e400", "float64", "overflow"),
        ("1e39", "float32", "overflow"),
        ("1e39", "float64", "rounded"),
        ("1e-2000", "float32", "underflow"),
        ("1e-2000", "float64", "underflow"),
        ("-1e-2000", "float32", "underflow"),
        ("1e-50", "float32", "underflow"),
        ("0.1", "float32", "rounded"),
        ("0.1", "float64", "rounded"),
        ("0.000000000931322574615478515625", "float32", "exact"),  # 2**-30
        ("0.000000000931322574615478515625", "float64", "exact"),
        ("16777217", "float32", "rounded"),
        ("16777217", "float64", "exact"),
        (16777217, "float32", "rounded"),
        ("inf", "float32", "special"),
        ("-inf", "float64", "special"),
        ("nan", "float32", "special"),
        ("nan", "float64", "special"),
        ("-0", "float32", "exact"),
        ("-0", "float64", "exact"),
        ("0", "float32", "exact"),
        ("10.625", "float32", "exact"),
        ("1e-45", "float32", "rounded"),
        (0.1, "float64", "exact"),  # a float is exactly itself in float64
        (0.1, "float32", "rounded"),
    ],
)
def test_describe_rounding_kind(value, fmt, kind):
    info = describe_rounding(value, encode(value, fmt))
    assert isinstance(info, RoundingInfo)
    assert info.kind == kind
    assert isinstance(info.message, str) and info.message


@pytest.mark.parametrize("value, fmt", [("0.1", "float32"), ("1e400", "float64"), ("1e-2000", "float32"),
                                        ("16777217", "float64"), ("-0", "float32")])
def test_describe_rounding_exact_value_is_typed_value(value, fmt):
    info = describe_rounding(value, encode(value, fmt))
    assert info.exact_value == Decimal(value)


@pytest.mark.parametrize("value", ["inf", "nan"])
def test_describe_rounding_special_has_no_exact_value(value):
    assert describe_rounding(value, encode(value)).exact_value is None


def test_describe_rounding_kind_agrees_with_exact_comparison_random():
    rng = random.Random(271828)
    for _ in range(400):
        digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 12)))
        text = "%s.%se%d" % (digits[0], digits[1:] or "0", rng.randint(-50, 40))
        q = Fraction(Decimal(text))
        for fmt in ("float32", "float64"):
            r = encode(text, fmt)
            if q == 0:
                expected = "exact"
            elif math.isinf(r.stored_value):
                expected = "overflow"
            elif r.stored_value == 0:
                expected = "underflow"
            else:
                expected = "exact" if Fraction(r.stored_value) == q else "rounded"
            assert describe_rounding(text, r).kind == expected, (text, fmt)
