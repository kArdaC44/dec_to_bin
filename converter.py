"""Exact decimal <-> binary conversion with step-by-step data.

The algorithms here are the manual ones taught in class:

* the integer part is converted by repeated division by 2 (the remainders,
  read bottom-to-top, are the bits);
* the fractional part is converted by repeated multiplication by 2 (the
  integer parts of the products, read top-to-bottom, are the bits).

All arithmetic uses :class:`fractions.Fraction`, so a string such as ``"0.1"``
means exactly one tenth, never the nearest float.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

DEFAULT_PRECISION = 16
MAX_PRECISION = 128
# Decimal/str inputs beyond 10**±MAX_DECIMAL_EXPONENT or with more significant
# digits than MAX_DECIMAL_DIGITS are rejected so conversions stay fast.
MAX_DECIMAL_EXPONENT = 10_000
MAX_DECIMAL_DIGITS = 10_000
# Integers wider than this have more than MAX_DECIMAL_DIGITS digits.
MAX_INTEGER_BITS = math.ceil(MAX_DECIMAL_DIGITS * math.log2(10))

_DECIMAL_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)")
_BINARY_RE = re.compile(r"([+-]?)(?:0[bB])?([01]*)(?:\.([01]*))?")


@dataclass(frozen=True)
class DivisionStep:
    """One division by 2 while converting the integer part."""

    dividend: int
    quotient: int
    remainder: int


@dataclass(frozen=True)
class MultiplicationStep:
    """One multiplication by 2 while converting the fractional part."""

    fraction: Fraction
    product: Fraction
    bit: int
    remaining: Fraction


@dataclass(frozen=True)
class ConversionResult:
    """Full result of a decimal -> binary conversion, including the steps."""

    binary: str
    negative: bool
    integer_part: int
    fraction_part: Fraction
    integer_steps: tuple[DivisionStep, ...]
    fraction_steps: tuple[MultiplicationStep, ...]
    is_exact: bool
    precision: int


def _validate_precision(precision: int) -> int:
    """Return ``precision`` if it is an int in ``1..MAX_PRECISION``."""
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise ValueError(f"Precision must be an integer, got {precision!r}.")
    if not 1 <= precision <= MAX_PRECISION:
        raise ValueError(
            f"Precision must be between 1 and {MAX_PRECISION}, got {precision}."
        )
    return precision


def _bounded_fraction(value: Decimal) -> Fraction:
    """Convert a finite Decimal to a Fraction, rejecting absurdly large inputs."""
    digits = value.as_tuple().digits
    significant = len(digits)
    while significant > 1 and digits[significant - 1] == 0:  # trailing zeros
        significant -= 1
    if value != 0 and (
        abs(value.adjusted()) > MAX_DECIMAL_EXPONENT
        or significant > MAX_DECIMAL_DIGITS
    ):
        raise ValueError(
            f"The number is too large, too small or too long to convert (limit: "
            f"magnitude 10^±{MAX_DECIMAL_EXPONENT}, {MAX_DECIMAL_DIGITS} digits)."
        )
    return Fraction(value)


def parse_decimal(value: int | float | Decimal | str) -> Fraction:
    """Parse ``value`` into an exact :class:`Fraction`.

    Strings must be plain decimal numbers (optional sign, digits, at most one
    ``.``). Floats are taken at their exact binary value. Booleans and
    non-finite numbers are rejected.

    Raises:
        ValueError: if the value is not a valid finite number.
    """
    if isinstance(value, bool):
        raise ValueError("Booleans are not accepted; enter a number.")
    if isinstance(value, int):
        if value.bit_length() > MAX_INTEGER_BITS:
            raise ValueError(
                f"The integer is too large to convert (limit: {MAX_DECIMAL_DIGITS} "
                "digits)."
            )
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot convert non-finite value {value!r}.")
        return Fraction(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError(f"Cannot convert non-finite value {value}.")
        return _bounded_fraction(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Please enter a number.")
        if not _DECIMAL_RE.fullmatch(text):
            raise ValueError(
                f"'{text}' is not a valid decimal number. Use digits with an "
                "optional sign and at most one '.', e.g. -10.625."
            )
        return _bounded_fraction(Decimal(text))
    raise ValueError(f"Unsupported input type: {type(value).__name__}.")


def fraction_to_decimal(value: Fraction) -> Decimal:
    """Convert a Fraction whose denominator is ``2**a * 5**b`` to an exact Decimal.

    Raises:
        ValueError: if the fraction has no finite decimal expansion.
    """
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1:
        raise ValueError(f"{value} has no finite decimal expansion.")
    scale = max(twos, fives)
    scaled = abs(value.numerator) * 10**scale // value.denominator
    digits = Decimal(scaled).as_tuple().digits
    sign = 1 if value < 0 else 0
    return Decimal((sign, digits, -scale))


def decimal_to_binary_steps(
    value: int | float | Decimal | str, precision: int = DEFAULT_PRECISION
) -> ConversionResult:
    """Convert a decimal number to binary, recording every step.

    Args:
        value: the number to convert (int, float, Decimal or decimal string).
        precision: maximum number of fractional bits (1..MAX_PRECISION). The
            fraction expansion is truncated, not rounded, at this length.

    Raises:
        ValueError: on invalid input or precision.
    """
    precision = _validate_precision(precision)
    number = parse_decimal(value)
    negative = number < 0
    magnitude = abs(number)
    integer_part = magnitude.numerator // magnitude.denominator
    fraction_part = magnitude - integer_part

    integer_steps: list[DivisionStep] = []
    dividend = integer_part
    while dividend > 0:
        quotient, remainder = divmod(dividend, 2)
        integer_steps.append(DivisionStep(dividend, quotient, remainder))
        dividend = quotient

    fraction_steps: list[MultiplicationStep] = []
    current = fraction_part
    while current != 0 and len(fraction_steps) < precision:
        product = current * 2
        bit = 1 if product >= 1 else 0
        remaining = product - bit
        fraction_steps.append(MultiplicationStep(current, product, bit, remaining))
        current = remaining

    int_bits = "".join(str(s.remainder) for s in reversed(integer_steps)) or "0"
    frac_bits = "".join(str(s.bit) for s in fraction_steps)
    binary = int_bits + (f".{frac_bits}" if frac_bits else "")
    if negative:
        binary = "-" + binary

    return ConversionResult(
        binary=binary,
        negative=negative,
        integer_part=integer_part,
        fraction_part=fraction_part,
        integer_steps=tuple(integer_steps),
        fraction_steps=tuple(fraction_steps),
        is_exact=current == 0,
        precision=precision,
    )


def decimal_to_binary(
    value: int | float | Decimal | str, precision: int = DEFAULT_PRECISION
) -> str:
    """Return the binary representation of ``value`` (e.g. ``"1010.101"``).

    See :func:`decimal_to_binary_steps` for details.
    """
    return decimal_to_binary_steps(value, precision).binary


def binary_to_decimal(binary: str) -> Decimal:
    """Convert a binary string such as ``"-0b1010.101"`` to its exact Decimal value.

    Raises:
        ValueError: if ``binary`` is not a valid binary number.
    """
    if not isinstance(binary, str):
        raise ValueError(f"Expected a string, got {type(binary).__name__}.")
    text = binary.strip()
    if not text:
        raise ValueError("Please enter a binary number.")
    match = _BINARY_RE.fullmatch(text)
    if match is None or not (match.group(2) or match.group(3)):
        raise ValueError(
            f"'{text}' is not a valid binary number. Use only 0 and 1 with an "
            "optional sign, optional 0b prefix and at most one '.', e.g. 1010.101."
        )
    sign, int_bits, frac_bits = match.group(1), match.group(2), match.group(3) or ""
    frac_bits = frac_bits.rstrip("0")
    numerator = int(int_bits + frac_bits or "0", 2)
    if sign == "-":
        numerator = -numerator
    return fraction_to_decimal(Fraction(numerator, 2 ** len(frac_bits)))
