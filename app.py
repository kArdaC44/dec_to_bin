"""Streamlit UI for the decimal <-> binary / IEEE 754 educational app.

All number logic lives in :mod:`converter` and :mod:`ieee754`; this module
only presents results.
"""

from __future__ import annotations

import html
import math
from decimal import Decimal
from fractions import Fraction

import streamlit as st

from converter import (
    DEFAULT_PRECISION,
    MAX_PRECISION,
    ConversionResult,
    binary_to_decimal,
    decimal_to_binary_steps,
    fraction_to_decimal,
)
from ieee754 import IEEE754Result, RoundingInfo, decode, describe_rounding, encode

MODE_D2B = "Decimal → Binary"
MODE_B2D = "Binary → Decimal"
MODE_IEEE = "IEEE 754 Visualizer"
MODES = (MODE_D2B, MODE_B2D, MODE_IEEE)
FORMAT_LABELS = {"Float32": "float32", "Float64": "float64"}
MAX_INPUT_CHARS = 400

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
.result-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1rem 0;
    background: rgba(128, 128, 128, 0.06);
}
.result-card .label { font-size: 0.85rem; opacity: 0.75; }
.result-card .value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.5rem; font-weight: 600; word-break: break-all;
}
.ieee-wrap { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0 1rem 0; }
.ieee-group { display: flex; flex-direction: column; gap: 0.25rem; }
.ieee-title { font-size: 0.8rem; font-weight: 600; letter-spacing: 0.03em; }
.ieee-bits { display: flex; flex-wrap: wrap; gap: 2px; }
.ieee-sub { font-size: 0.75rem; opacity: 0.7; }
.bit {
    display: inline-block; min-width: 1.35rem; text-align: center;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-weight: 600; padding: 0.2rem 0.1rem; border-radius: 4px;
    border: 1px solid transparent;
}
.bit.sign { background: rgba(239, 68, 68, 0.20); border-color: rgba(239, 68, 68, 0.65); }
.bit.exp  { background: rgba(59, 130, 246, 0.20); border-color: rgba(59, 130, 246, 0.65); }
.bit.man  { background: rgba(34, 197, 94, 0.18); border-color: rgba(34, 197, 94, 0.60); }
.ieee-title.sign { color: rgb(239, 68, 68); }
.ieee-title.exp  { color: rgb(59, 130, 246); }
.ieee-title.man  { color: rgb(34, 197, 94); }
</style>
"""

CATEGORY_TEXT = {
    "zero": (
        "**Zero.** Exponent and mantissa are all 0s. IEEE 754 has both +0 and "
        "−0; they differ only in the sign bit and compare equal."
    ),
    "subnormal": (
        "**Subnormal (denormal).** The exponent field is all 0s but the "
        "mantissa is not. There is no implicit leading 1: the value is "
        "(−1)^s × 0.mantissa × 2^(1 − bias). Subnormals fill the gap between "
        "zero and the smallest normal number, at the cost of precision."
    ),
    "normal": (
        "**Normal number.** The value is (−1)^s × 1.mantissa × 2^(exponent − "
        "bias). The leading 1 is implicit and not stored, which gains one bit "
        "of precision for free."
    ),
    "infinity": (
        "**Infinity.** Exponent all 1s and mantissa all 0s. Produced by "
        "overflow (values too large for the format) or by entering inf."
    ),
    "nan": (
        "**NaN (Not a Number).** Exponent all 1s and a non-zero mantissa. "
        "Represents undefined results such as 0/0. NaN is not equal to "
        "anything, including itself."
    ),
}


def fraction_text(value: Fraction) -> str:
    """Render a Fraction as an exact decimal string (fallback: p/q)."""
    try:
        return format(fraction_to_decimal(value), "f")
    except ValueError:
        return f"{value.numerator}/{value.denominator}"


def result_card(label: str, value: str) -> None:
    """Show a highlighted result box; both strings are HTML-escaped."""
    st.markdown(
        f'<div class="result-card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def input_form(mode: str, label: str, placeholder: str, help_text: str) -> str | None:
    """Show the input form for ``mode`` and return the last submitted value."""
    state_key = f"submitted::{mode}"
    with st.form(key=f"form::{mode}"):
        text = st.text_input(
            label,
            placeholder=placeholder,
            help=help_text,
            max_chars=MAX_INPUT_CHARS,
            key=f"input::{mode}",
        )
        if st.form_submit_button("Convert", type="primary"):
            st.session_state[state_key] = text
    return st.session_state.get(state_key)


# --------------------------------------------------------------------------- #
# Decimal -> Binary
# --------------------------------------------------------------------------- #


def show_division_steps(result: ConversionResult) -> None:
    """Explain the integer part via repeated division by 2."""
    st.markdown(f"#### Integer part: {result.integer_part}")
    if not result.integer_steps:
        st.info("The integer part is 0, so its binary form is simply **0**.")
        return
    st.markdown(
        "Divide by 2 repeatedly and record each remainder. Stop when the "
        "quotient reaches 0. The bits are the remainders **read from bottom "
        "to top** (the last remainder is the most significant bit)."
    )
    rows = [
        {
            "Step": i,
            "Dividend": str(s.dividend),
            "÷ 2 = Quotient": str(s.quotient),
            "Remainder (bit)": s.remainder,
        }
        for i, s in enumerate(result.integer_steps, start=1)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    bits = "".join(str(s.remainder) for s in reversed(result.integer_steps))
    st.markdown(f"Remainders bottom → top: `{bits}`")


def show_multiplication_steps(result: ConversionResult) -> None:
    """Explain the fractional part via repeated multiplication by 2."""
    st.markdown(f"#### Fractional part: {fraction_text(result.fraction_part)}")
    if not result.fraction_steps:
        st.info("There is no fractional part, so no bits follow the binary point.")
        return
    st.markdown(
        "Multiply the fraction by 2. The integer part of the product (0 or 1) "
        "is the next bit; keep only the fractional remainder and repeat. The "
        "bits are read **from top to bottom**."
    )
    rows = [
        {
            "Step": i,
            "Fraction": fraction_text(s.fraction),
            "× 2 =": fraction_text(s.product),
            "Bit": s.bit,
            "Remaining": fraction_text(s.remaining),
        }
        for i, s in enumerate(result.fraction_steps, start=1)
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    bits = "".join(str(s.bit) for s in result.fraction_steps)
    st.markdown(f"Bits top → bottom: `.{bits}`")
    if result.is_exact:
        st.success("The remaining fraction reached 0, so this expansion is exact.")
    else:
        st.warning(
            f"The remaining fraction never reached 0 within {result.precision} "
            "bits, so the result was **truncated** (not rounded). A fraction "
            "terminates in binary only if its denominator (in lowest terms) is "
            "a power of 2 — e.g. 0.1 = 1/10 repeats forever "
            "(0.000110011001100…). Increase the precision in the sidebar to "
            "see more bits."
        )


def render_decimal_to_binary(precision: int) -> None:
    """Decimal → Binary mode."""
    st.subheader("Decimal → Binary")
    value = input_form(
        MODE_D2B,
        "Decimal number",
        "e.g. 10.625, -0.1, 255",
        "Digits with an optional sign and at most one decimal point.",
    )
    if value is None:
        return
    try:
        result = decimal_to_binary_steps(value, precision)
    except ValueError as exc:
        st.error(str(exc))
        return

    result_card(f"{value.strip()} in binary", result.binary)
    st.caption("Copy-ready output:")
    st.code(result.binary, language=None)
    if not result.is_exact:
        st.caption(f"Truncated to {precision} fractional bits.")

    st.markdown("### Step-by-step explanation")
    if result.negative:
        st.markdown(
            "The number is negative: convert its absolute value, then put a "
            "**−** sign in front (sign-magnitude notation)."
        )
    left, right = st.columns(2, gap="large")
    with left:
        show_division_steps(result)
    with right:
        show_multiplication_steps(result)


# --------------------------------------------------------------------------- #
# Binary -> Decimal
# --------------------------------------------------------------------------- #


def render_binary_to_decimal() -> None:
    """Binary → Decimal mode."""
    st.subheader("Binary → Decimal")
    value = input_form(
        MODE_B2D,
        "Binary number",
        "e.g. 1010.101, -0b1101, 0.0001",
        "0s and 1s with an optional sign, optional 0b prefix and at most one '.'.",
    )
    if value is None:
        return
    try:
        result = binary_to_decimal(value)
    except ValueError as exc:
        st.error(str(exc))
        return

    result_text = format(result, "f")
    result_card(f"{value.strip()} in decimal", result_text)
    st.code(result_text, language=None, wrap_lines=True)

    st.markdown("### Step-by-step explanation")
    text = value.strip().lstrip("+-")
    if text[:2].lower() == "0b":
        text = text[2:]
    int_bits, _, frac_bits = text.partition(".")
    st.markdown(
        "Each bit is multiplied by a power of 2 given by its position: "
        "2⁰, 2¹, 2², … to the left of the point and 2⁻¹, 2⁻², … to the right. "
        "Adding the contributions of the 1-bits gives the value."
    )
    rows = []
    for i, bit in enumerate(int_bits):
        rows.append((bit, len(int_bits) - 1 - i))
    for i, bit in enumerate(frac_bits):
        rows.append((bit, -(i + 1)))
    table = [
        {
            "Bit": int(bit),
            "Position": f"2^{power}",
            "Place value": fraction_text(Fraction(2) ** power),
            "Contribution": fraction_text(int(bit) * Fraction(2) ** power),
        }
        for bit, power in rows
    ]
    if len(table) <= 256:
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("The input is long; the per-bit table is omitted.")
    sign = "−" if value.strip().startswith("-") and result != 0 else ""
    st.markdown(f"Sum of contributions = {sign}**{format(abs(result), 'f')}**")


# --------------------------------------------------------------------------- #
# IEEE 754
# --------------------------------------------------------------------------- #


def bit_group_html(title: str, css: str, bits: str, subtitle: str) -> str:
    """Return HTML for one colored group of bits."""
    spans = "".join(f'<span class="bit {css}">{html.escape(b)}</span>' for b in bits)
    return (
        f'<div class="ieee-group"><div class="ieee-title {css}">{html.escape(title)}</div>'
        f'<div class="ieee-bits">{spans}</div>'
        f'<div class="ieee-sub">{html.escape(subtitle)}</div></div>'
    )


def show_bit_layout(r: IEEE754Result) -> None:
    """Render sign / exponent / mantissa bits as colored boxes."""
    groups = [
        bit_group_html("SIGN", "sign", r.sign_bit, "1 bit"),
        bit_group_html("EXPONENT", "exp", r.exponent_bits, f"{r.fmt.exponent_bits} bits"),
        bit_group_html("MANTISSA (FRACTION)", "man", r.mantissa_bits, f"{r.fmt.mantissa_bits} bits"),
    ]
    st.markdown(f'<div class="ieee-wrap">{"".join(groups)}</div>', unsafe_allow_html=True)


def float_text(value: float) -> str:
    """Human-readable repr of a float (handles inf/nan)."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "-inf" if value < 0 else "inf"
    return repr(value)


def show_math_vs_ieee(info: RoundingInfo, r: IEEE754Result, precision: int) -> None:
    """Contrast the plain binary expansion with the IEEE 754 encoding."""
    st.markdown("### Mathematical binary vs. IEEE 754 encoding")
    if info.kind == "special":
        st.info(
            "Infinity and NaN have no ordinary binary expansion — they exist "
            "only as special bit patterns in IEEE 754."
        )
        return
    if info.exact_value is None:
        st.info("The number is too large to show its mathematical binary expansion.")
        return
    try:
        math_result = decimal_to_binary_steps(info.exact_value, precision)
    except ValueError as exc:
        st.info(f"The mathematical binary expansion is not shown: {exc}")
        return
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Mathematical binary** (exact value you typed)")
        st.code(
            math_result.binary + ("…" if not math_result.is_exact else ""),
            language=None,
            wrap_lines=True,
        )
        st.caption(
            "A plain positional number: sign, integer bits, binary point, "
            "fraction bits. It may need any number of bits"
            + ("" if math_result.is_exact else f" (shown truncated to {precision})")
            + "."
        )
    with right:
        st.markdown(f"**IEEE 754 {r.fmt.name} encoding** (what the computer stores)")
        st.code(f"{r.sign_bit} {r.exponent_bits} {r.mantissa_bits}", language=None)
        st.caption(
            f"A fixed-size {r.fmt.total_bits}-bit pattern: the number is "
            "normalized to 1.xxx × 2^e (scientific notation in base 2); the "
            "sign, the biased exponent and the bits after the leading 1 are "
            "stored separately. Bits beyond the mantissa width are rounded away."
        )
    if r.category == "normal":
        st.markdown(
            f"Normalized form: **(−1)^{r.sign_bit} × 1.{r.mantissa_bits.rstrip('0') or '0'}"
            f" × 2^{r.actual_exponent}** — the exponent field stores "
            f"{r.actual_exponent} + {r.bias} = {r.raw_exponent}."
        )
    elif r.category == "subnormal":
        st.markdown(
            f"Subnormal form: **(−1)^{r.sign_bit} × 0.{r.mantissa_bits.rstrip('0')}"
            f" × 2^{r.actual_exponent}**."
        )


def render_ieee754(precision: int, fmt: str) -> None:
    """IEEE 754 Visualizer mode."""
    st.subheader(f"IEEE 754 Visualizer — {fmt}")
    value = input_form(
        MODE_IEEE,
        "Number",
        "e.g. 10, 0.1, -0, 1e-45, inf, nan",
        "Any number Python's float() accepts, including inf, -inf, nan and 1e-45.",
    )
    if value is None:
        return
    try:
        r = encode(value, fmt)
        info = describe_rounding(value, r)
    except ValueError as exc:
        st.error(str(exc))
        return

    result_card(f"{value.strip()} as IEEE 754 {fmt}", r.hex)
    show_bit_layout(r)
    st.code(r.bits, language=None)

    reconstructed = decode(r.bits, fmt)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Fields")
        rows = [
            ("Sign bit", f"{r.sign_bit} ({'negative' if r.is_negative else 'positive'})"),
            ("Exponent bits", r.exponent_bits),
            ("Mantissa bits", r.mantissa_bits),
            ("Bias", str(r.bias)),
            ("Raw exponent (unsigned)", str(r.raw_exponent)),
            (
                "Actual exponent",
                "— (special value)" if r.actual_exponent is None
                else f"{r.actual_exponent}"
                + (f" = {r.raw_exponent} − {r.bias}" if r.category == "normal"
                   else f" = 1 − {r.bias} (subnormal)" if r.category == "subnormal" else ""),
            ),
            ("Category", r.category),
            ("Hexadecimal", r.hex),
            ("Reconstructed value", float_text(reconstructed)),
        ]
        st.dataframe(
            [{"Field": k, "Value": v} for k, v in rows], hide_index=True, width="stretch"
        )
    with right:
        st.markdown("#### What this means")
        st.markdown(CATEGORY_TEXT[r.category])
        if math.isfinite(r.stored_value):
            st.markdown("Exact value stored:")
            st.code(format(Decimal(r.stored_value), "f"), language=None, wrap_lines=True)

    if info.kind in ("rounded", "overflow", "underflow"):
        st.warning(info.message)
    elif info.kind == "exact":
        st.success(info.message)

    show_math_vs_ieee(info, r, precision)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="Decimal ↔ Binary", page_icon="🔢", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Conversion mode", MODES)
        precision = st.slider(
            "Fractional precision (bits)",
            min_value=1,
            max_value=MAX_PRECISION,
            value=DEFAULT_PRECISION,
            help="Maximum number of bits after the binary point. Longer "
            "expansions are truncated.",
        )
        fmt_label = st.radio("IEEE 754 format", list(FORMAT_LABELS), horizontal=True)
        st.divider()
        st.caption(
            "Float32: 1 sign + 8 exponent + 23 mantissa bits (bias 127).  \n"
            "Float64: 1 sign + 11 exponent + 52 mantissa bits (bias 1023)."
        )

    st.title("Decimal ↔ Binary Converter")
    st.markdown(
        "Learn how numbers are written in base 2 — and how computers actually "
        "store them using the IEEE 754 floating-point standard."
    )

    if mode == MODE_D2B:
        render_decimal_to_binary(precision)
    elif mode == MODE_B2D:
        render_binary_to_decimal()
    else:
        render_ieee754(precision, FORMAT_LABELS[fmt_label])


main()
