"""Smoke tests for app.py using Streamlit's AppTest harness."""
import os

import pytest

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

D2B, B2D, IEEE = 0, 1, 2  # indexes into the "Conversion mode" radio options


def make_app(mode, fmt="Float32"):
    at = AppTest.from_file(APP, default_timeout=60).run()
    mode_radio = at.sidebar.radio[0]
    mode_radio.set_value(mode_radio.options[mode])
    at.sidebar.radio[1].set_value(fmt)
    return at.run()


def submit(at, value):
    at.text_input[0].input(value)
    return at.button[0].click().run()


def page_text(at):
    parts = []
    for elements in (at.markdown, at.code, at.caption, at.success, at.warning, at.info, at.error):
        parts.extend(str(e.value) for e in elements)
    return "\n".join(parts)


def test_app_loads_without_exception():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [r.label for r in at.sidebar.radio] == ["Conversion mode", "IEEE 754 format"]


CASES = {
    D2B: ["10.625", "-0.1", "0", "255", "abc", "1.2.3", "١٠", "1e3", "9" * 399],
    B2D: ["1010.101", "-0b1101", "0.000000000000000000000000000001", "102", "1.0.1", "0b", "1" * 399],
    IEEE: ["10", "0.1", "-0", "1e-45", "inf", "-inf", "nan", "1e400", "1e-2000", "16777217", "abc", "0x10"],
}


@pytest.mark.parametrize("fmt", ["Float32", "Float64"])
@pytest.mark.parametrize(
    "mode, value",
    [(m, v) for m, values in CASES.items() for v in values],
)
def test_every_mode_and_format_runs_without_exception(mode, value, fmt):
    at = submit(make_app(mode, fmt), value)
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("mode", [D2B, B2D, IEEE])
def test_empty_submit_shows_error(mode):
    at = submit(make_app(mode), "")
    assert not at.exception
    assert len(at.error) >= 1


@pytest.mark.parametrize("mode, value", [(D2B, "abc"), (D2B, "1,5"), (B2D, "102"), (B2D, "1.0.1"),
                                         (IEEE, "abc"), (IEEE, "1.2.3")])
def test_invalid_input_shows_error(mode, value):
    at = submit(make_app(mode), value)
    assert not at.exception
    assert len(at.error) >= 1


@pytest.mark.parametrize("mode, value", [(D2B, "10.625"), (B2D, "1010.101"), (IEEE, "10")])
def test_valid_input_shows_no_error(mode, value):
    at = submit(make_app(mode), value)
    assert not at.exception
    assert len(at.error) == 0


def test_decimal_to_binary_shows_result():
    at = submit(make_app(D2B), "10.625")
    assert "1010.101" in [c.value for c in at.code]


def test_binary_to_decimal_tiny_value_shown_in_plain_notation():
    at = submit(make_app(B2D), "0.000000000000000000000000000001")
    text = page_text(at)
    assert "0.000000000931322574615478515625" in text
    assert "E-" not in text


@pytest.mark.parametrize("fmt", ["Float32", "Float64"])
def test_ieee_overflow_shows_warning(fmt):
    at = submit(make_app(IEEE, fmt), "1e400")
    assert not at.exception
    assert any("overflow" in w.value.lower() for w in at.warning)


def test_ieee_shows_reference_hex_for_ten():
    at = submit(make_app(IEEE, "Float32"), "10")
    assert "0x41200000" in page_text(at)


def test_ieee_float64_point_one_hex():
    at = submit(make_app(IEEE, "Float64"), "0.1")
    assert "0x3fb999999999999a" in page_text(at)
