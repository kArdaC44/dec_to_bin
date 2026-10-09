# Decimal ↔ Binary Converter

An educational Streamlit app that shows **how** numbers are converted between
decimal and binary, and how computers actually **store** them using the
IEEE 754 floating-point standard.

## Features

- **Decimal → Binary** using the manual classroom algorithms:
  - integer part by repeated division by 2 (remainders read bottom-to-top),
  - fractional part by repeated multiplication by 2 (bits read top-to-bottom),
  - exact arithmetic (`fractions.Fraction`), so `0.1` really means 1/10,
  - configurable fractional precision (1–128 bits) with a clear note when the
    expansion is truncated.
- **Binary → Decimal** with an exact `Decimal` result (no rounding, even for
  very long inputs) and a per-bit place-value table.
- **IEEE 754 Visualizer** for Float32 and Float64:
  - color-coded sign / exponent / mantissa bits,
  - bias, raw and actual exponent, category, hex, reconstructed value,
  - the exact value stored and a note when rounding or overflow occurred,
  - explanations of special values: ±0, ±infinity, NaN and subnormals.
- Copy-ready output, input validation with friendly error messages, and
  layout that works in light and dark themes.

## Project structure

```
decimal_to_binary/
├── app.py            # Streamlit UI (presentation only)
├── converter.py      # Exact decimal <-> binary conversion + step data
├── ieee754.py        # IEEE 754 float32/float64 encode, decode, field split
├── requirements.txt
├── README.md
└── tests/
    ├── conftest.py
    ├── test_app.py
    ├── test_converter.py
    └── test_ieee754.py
```

## Installation

Requires Python 3.11+.

```bash
python -m venv .venv
```

Activate the virtual environment:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **macOS / Linux:** `source .venv/bin/activate`

Then install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the app

```bash
streamlit run app.py
```

Streamlit opens the app in your browser (by default at http://localhost:8501).
Choose a mode in the sidebar, type a number and press **Convert** (or Enter).

## Running the tests

```bash
python -m pytest
```

## Using the modules directly

```python
from converter import decimal_to_binary, binary_to_decimal
from ieee754 import encode, decode

decimal_to_binary("10.625")          # '1010.101'
decimal_to_binary("0.1", 10)         # '0.0001100110' (truncated)
binary_to_decimal("1010.101")        # Decimal('10.625')
encode(10.0, "float32").hex          # '0x41200000'
decode(encode(0.1, "float32").bits)  # 0.10000000149011612
```

## Mathematical binary vs. IEEE 754

These are two different things that are easy to confuse.

**Mathematical (positional) binary** is just a number written in base 2:
`10.625` is `1010.101` because
`1·2³ + 0·2² + 1·2¹ + 0·2⁰ + 1·2⁻¹ + 0·2⁻² + 1·2⁻³ = 10.625`.
It can use as many bits as needed, and some numbers never end: `0.1` is
`0.000110011001100…` repeating forever, because only fractions whose
denominator is a power of 2 terminate in binary.

**IEEE 754** is a fixed-size *encoding* that computers use to store such
numbers. The value is first normalized to binary scientific notation,
`1010.101 = 1.010101 × 2³`, and then three fields are stored:

| Field    | Float32 | Float64 | Contents                                         |
|----------|---------|---------|--------------------------------------------------|
| Sign     | 1 bit   | 1 bit   | 0 = positive, 1 = negative                       |
| Exponent | 8 bits  | 11 bits | actual exponent + bias (127 / 1023)              |
| Mantissa | 23 bits | 52 bits | bits after the implicit leading `1.`             |

So `10.625` in Float32 is `0 10000010 01010100000000000000000`
(`0x412a0000`): sign 0, exponent 3 + 127 = 130, mantissa `010101` padded
with zeros. Because the mantissa has a fixed width, values like `0.1` must be
**rounded** — Float32 actually stores `0.100000001490116119384765625`.

Special bit patterns:

- **±0** — exponent and mantissa all 0 (the sign bit distinguishes −0).
- **Subnormal** — exponent all 0, mantissa non-zero: `0.m × 2^(1 − bias)`,
  extending the range toward zero with reduced precision.
- **±Infinity** — exponent all 1, mantissa 0 (also the result of overflow).
- **NaN** — exponent all 1, mantissa non-zero.
