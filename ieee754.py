"""IEEE 754 binary32 / binary64 encoding, field splitting and decoding."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction

# Decimal magnitudes beyond 10**±_EXPONENT_LIMIT overflow / underflow every
# supported format, so they are classified without building huge Fractions.
# (converter.MAX_DECIMAL_EXPONENT is the analogous, larger guard for the
# exact binary expansion, which has no format range to clamp to.)
_EXPONENT_LIMIT = 1000
# Integers wider than this overflow every supported format (float64 max < 2**1024);
# they are handled without the (slow) int -> Decimal conversion.
_INT_BIT_LIMIT = 1100


@dataclass(frozen=True)
class FloatFormat:
    """Layout of an IEEE 754 binary floating-point format."""

    name: str
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    struct_code: str


FORMATS: dict[str, FloatFormat] = {
    "float32": FloatFormat(
        name="float32", total_bits=32, exponent_bits=8, mantissa_bits=23,
        bias=127, struct_code="f",
    ),
    "float64": FloatFormat(
        name="float64", total_bits=64, exponent_bits=11, mantissa_bits=52,
        bias=1023, struct_code="d",
    ),
}


@dataclass(frozen=True)
class IEEE754Result:
    """An encoded value split into its IEEE 754 fields."""

    fmt: FloatFormat
    input_value: float
    bits: str
    sign_bit: str
    exponent_bits: str
    mantissa_bits: str
    hex: str
    bias: int
    raw_exponent: int
    actual_exponent: int | None
    category: str
    stored_value: float
    is_negative: bool


def get_format(fmt: str | FloatFormat) -> FloatFormat:
    """Return the :class:`FloatFormat` named ``fmt`` (or ``fmt`` itself).

    Raises:
        ValueError: if the format is unknown.
    """
    if isinstance(fmt, FloatFormat):
        return fmt
    if isinstance(fmt, str) and fmt.lower() in FORMATS:
        return FORMATS[fmt.lower()]
    raise ValueError(f"Unknown format {fmt!r}; choose one of {sorted(FORMATS)}.")


def parse_float(value: float | int | str) -> float:
    """Parse ``value`` into a Python float (binary64).

    Strings accept anything :func:`float` accepts, e.g. ``"inf"``, ``"nan"``,
    ``"-0"`` or ``"1e-45"``.

    Raises:
        ValueError: on invalid input.
    """
    if isinstance(value, bool):
        raise ValueError("Booleans are not accepted; enter a number.")
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        try:
            return float(value)
        except OverflowError:
            return math.inf if value > 0 else -math.inf
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Please enter a number.")
        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f"'{text}' is not a valid number. Examples: 10.5, -0.1, 1e-45, "
                "inf, nan."
            ) from None
    raise ValueError(f"Unsupported input type: {type(value).__name__}.")


@dataclass(frozen=True)
class RoundingInfo:
    """How the typed value relates to the value actually stored.

    ``kind`` is one of ``"exact"``, ``"rounded"``, ``"overflow"``,
    ``"underflow"`` or ``"special"`` (literal inf / nan input).
    ``exact_value`` is the exact typed value, or None for inf / nan and for
    integers wider than ``_INT_BIT_LIMIT`` bits (which always overflow).
    """

    kind: str
    message: str
    exact_value: Decimal | None


def exact_value(value: float | int | str) -> Decimal | None:
    """Return the exact value of ``value`` as a Decimal, or None if non-finite.

    Strings are read exactly (``"0.1"`` is one tenth, not the nearest float).
    Integers wider than ``_INT_BIT_LIMIT`` bits also return None: they overflow
    every supported format and converting them to Decimal would be slow.

    Raises:
        ValueError: on invalid input.
    """
    number = parse_float(value)  # validates the input
    if isinstance(value, int):
        return Decimal(value) if value.bit_length() <= _INT_BIT_LIMIT else None
    if isinstance(value, str):
        try:
            exact = Decimal(value.strip())
        except InvalidOperation:  # float() syntax Decimal rejects
            exact = Decimal(number) if math.isfinite(number) else None
        return exact if exact is not None and exact.is_finite() else None
    return Decimal(number) if math.isfinite(number) else None


def _round_exact(value: Decimal, layout: FloatFormat) -> float:
    """Round an exact decimal to ``layout`` (round-half-even, subnormals, overflow).

    The result is returned as a Python float, which holds it exactly for
    formats no wider than binary64.
    """
    sign = -1.0 if value.is_signed() else 1.0
    if value == 0 or value.adjusted() < -_EXPONENT_LIMIT:
        return math.copysign(0.0, sign)
    if value.adjusted() > _EXPONENT_LIMIT:
        return math.copysign(math.inf, sign)
    q = abs(Fraction(value))
    m = layout.mantissa_bits
    # Find e with 2**e <= q < 2**(e + 1), clamped to the subnormal range.
    e = q.numerator.bit_length() - q.denominator.bit_length()
    while Fraction(2) ** e > q:
        e -= 1
    while Fraction(2) ** (e + 1) <= q:
        e += 1
    e = max(e, 1 - layout.bias)
    scaled = q * Fraction(2) ** (m - e)
    significand, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (
        twice == scaled.denominator and significand % 2 == 1
    ):
        significand += 1
    if significand == 1 << (m + 1):
        significand >>= 1
        e += 1
    if e > layout.bias:
        return math.copysign(math.inf, sign)
    return math.copysign(math.ldexp(significand, e - m), sign)


def _shown(value: float | int | str) -> str:
    """Return a short display form of ``value`` for messages."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and value.bit_length() > 166:  # more than ~50 digits
        # Estimate the decimal exponent from the top bits; str() of a huge int
        # is slow and may exceed Python's int-to-str digit limit.
        shift = value.bit_length() - 64
        log10 = math.log10(abs(value) >> shift) + shift * math.log10(2)
        exponent = math.floor(log10)
        sign = "-" if value < 0 else ""
        return f"{sign}{10 ** (log10 - exponent):.6f}e+{exponent}"
    return repr(value)


def describe_rounding(value: float | int | str, result: IEEE754Result) -> RoundingInfo:
    """Compare the typed ``value`` with ``result.stored_value``.

    Raises:
        ValueError: on invalid input.
    """
    exact = exact_value(value)
    shown = _shown(value)
    stored = result.stored_value
    name = result.fmt.name
    if exact is None and isinstance(value, int):  # wider than _INT_BIT_LIMIT bits
        return RoundingInfo(
            "overflow",
            f"Overflow: {shown} is larger in magnitude than the largest finite "
            f"{name} value, so it is stored as {'-' if value < 0 else ''}infinity.",
            None,
        )
    if exact is None:
        return RoundingInfo(
            "special",
            f"{shown} is a special value; IEEE 754 encodes it with a reserved "
            "bit pattern, not as an ordinary number.",
            None,
        )
    if exact == 0:
        return RoundingInfo("exact", f"Zero is represented exactly in {name}.", exact)
    if math.isinf(stored):
        return RoundingInfo(
            "overflow",
            f"Overflow: {shown} is larger in magnitude than the largest finite "
            f"{name} value, so it is stored as {'-' if stored < 0 else ''}infinity.",
            exact,
        )
    if stored == 0:
        return RoundingInfo(
            "underflow",
            f"Underflow: {shown} is closer to zero than to the smallest {name} "
            "subnormal, so it is stored as zero.",
            exact,
        )
    if Fraction(stored) == Fraction(exact):
        return RoundingInfo(
            "exact", f"{shown} is represented exactly in {name}.", exact
        )
    return RoundingInfo(
        "rounded",
        f"Rounding: {shown} cannot be represented exactly in {name}; the nearest "
        f"value {stored!r} is stored instead.",
        exact,
    )


def _category(fmt: FloatFormat, raw_exponent: int, mantissa: int) -> str:
    """Classify a value from its raw exponent and mantissa fields."""
    if raw_exponent == (1 << fmt.exponent_bits) - 1:
        return "infinity" if mantissa == 0 else "nan"
    if raw_exponent == 0:
        return "zero" if mantissa == 0 else "subnormal"
    return "normal"


def encode(value: float | int | str, fmt: str | FloatFormat = "float32") -> IEEE754Result:
    """Encode ``value`` in the IEEE 754 format ``fmt`` and split its fields.

    For float32, finite values too large to represent round to +/-infinity
    (IEEE round-to-nearest overflow) instead of raising.

    Raises:
        ValueError: on invalid input or unknown format.
    """
    layout = get_format(fmt)
    number = parse_float(value)
    target = number
    if layout.mantissa_bits < FORMATS["float64"].mantissa_bits and not isinstance(
        value, float
    ):
        # Round the exact typed value once, avoiding double rounding through float64.
        exact = exact_value(value)
        if exact is not None:
            target = _round_exact(exact, layout)
    code = ">" + layout.struct_code
    try:
        packed = struct.pack(code, target)
    except OverflowError:
        packed = struct.pack(code, math.copysign(math.inf, target))
    stored_value = struct.unpack(code, packed)[0]

    as_int = int.from_bytes(packed, "big")
    bits = format(as_int, f"0{layout.total_bits}b")
    sign_bit = bits[0]
    exponent_bits = bits[1 : 1 + layout.exponent_bits]
    mantissa_bits = bits[1 + layout.exponent_bits :]
    raw_exponent = int(exponent_bits, 2)
    category = _category(layout, raw_exponent, int(mantissa_bits, 2))
    if category == "normal":
        actual_exponent: int | None = raw_exponent - layout.bias
    elif category == "subnormal":
        actual_exponent = 1 - layout.bias
    else:
        actual_exponent = None

    return IEEE754Result(
        fmt=layout,
        input_value=number,
        bits=bits,
        sign_bit=sign_bit,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        hex=f"0x{as_int:0{layout.total_bits // 4}x}",
        bias=layout.bias,
        raw_exponent=raw_exponent,
        actual_exponent=actual_exponent,
        category=category,
        stored_value=stored_value,
        is_negative=sign_bit == "1",
    )


def decode(bits: str, fmt: str | FloatFormat = "float32") -> float:
    """Reconstruct a value mathematically from its IEEE 754 bit pattern.

    Normal: ``(-1)^s * 1.m * 2^(e - bias)``; subnormal:
    ``(-1)^s * 0.m * 2^(1 - bias)``; all-ones exponent gives +/-inf or nan.
    Whitespace in ``bits`` is ignored.

    Raises:
        ValueError: on wrong length, non-binary characters or unknown format.
    """
    layout = get_format(fmt)
    if not isinstance(bits, str):
        raise ValueError(f"Expected a bit string, got {type(bits).__name__}.")
    clean = "".join(bits.split())
    if len(clean) != layout.total_bits or set(clean) - {"0", "1"}:
        raise ValueError(
            f"Expected exactly {layout.total_bits} bits of 0/1 for "
            f"{layout.name}, got {clean!r}."
        )
    sign = -1.0 if clean[0] == "1" else 1.0
    raw_exponent = int(clean[1 : 1 + layout.exponent_bits], 2)
    mantissa = int(clean[1 + layout.exponent_bits :], 2)
    category = _category(layout, raw_exponent, mantissa)

    if category == "nan":
        return math.nan
    if category == "infinity":
        return sign * math.inf
    if category in ("zero", "subnormal"):
        # 0.m * 2^(1 - bias) == m * 2^(1 - bias - mantissa_bits)
        magnitude = math.ldexp(mantissa, 1 - layout.bias - layout.mantissa_bits)
    else:
        # 1.m * 2^(e - bias) == (2^mantissa_bits + m) * 2^(e - bias - mantissa_bits)
        significand = (1 << layout.mantissa_bits) | mantissa
        magnitude = math.ldexp(
            significand, raw_exponent - layout.bias - layout.mantissa_bits
        )
    return math.copysign(magnitude, sign)
