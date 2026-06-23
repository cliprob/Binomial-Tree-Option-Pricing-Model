# Binomial Tree Option Pricing Model

## Overview

This project implements the Cox-Ross-Rubinstein (CRR) binomial tree model
for European and American option pricing. It uses risk-neutral pricing,
supports early exercise for American options, and provides a vectorized NumPy
backward induction implementation alongside a loop-based reference
implementation.

The European option outputs are validated against the Black-Scholes
closed-form model, and the repository includes a benchmark script comparing
the vectorized implementation with the loop-based reference.

The original notebook is preserved in `notebooks/` for traceability. The
active implementation lives in `src/option_pricing/`.

## Why This Project Matters

This repository demonstrates:

- derivatives pricing fundamentals,
- numerical methods for option valuation,
- clean Python package structure,
- NumPy vectorization,
- unit testing for quantitative code.

## Methodology

The CRR model discretizes the underlying asset price into an up/down tree.
For each time step, the model computes:

- up factor: `u = exp(sigma * sqrt(dt))`,
- down factor: `d = 1 / u`,
- risk-neutral probability:
  `p = (exp(r * dt) - d) / (u - d)`.

At maturity, option values are initialized from terminal payoffs. Prices are
then computed by backward induction using the continuously compounded
discount factor `exp(-r * dt)`. For American options, the continuation value is
compared with the immediate exercise value at every node.

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Usage

```python
from option_pricing import binomial_tree_price_vectorized

price = binomial_tree_price_vectorized(
    s0=40,
    k=45,
    r=0.03,
    sigma=0.25,
    T=1.0,
    n_steps=100,
    option_type="put",
    exercise="american",
)

print(round(price, 4))
```

## Example Prices

Prices below were generated from the implemented package using
`N=100`, `S0=40`, `K=45`, `T=1`, `r=3%`, and `sigma=25%`.

| Option        | Parameters                               |  Price |
| ------------- | ---------------------------------------- | -----: |
| European Call | S0=40, K=45, T=1, r=3%, sigma=25%, N=100 | 2.5857 |
| European Put  | S0=40, K=45, T=1, r=3%, sigma=25%, N=100 | 6.2558 |
| American Call | S0=40, K=45, T=1, r=3%, sigma=25%, N=100 | 2.5857 |
| American Put  | S0=40, K=45, T=1, r=3%, sigma=25%, N=100 | 6.5252 |

## Validation

The test suite checks that:

- vectorized and loop implementations agree,
- European prices converge to Black-Scholes prices,
- put-call parity approximately holds for European options,
- American put value is at least the European put value,
- American call value without dividends is approximately equal to the
  European call value,
- invalid inputs raise `ValueError`.

## Benchmark

Run the local benchmark with:

```bash
make benchmark
```

Example local benchmark output is shown below. Timings depend on hardware,
Python version, and current system load, so they should be read as a local
comparison rather than a universal performance claim.

| N | Loop price | Vectorized price | Loop time (ms) | Vectorized time (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 100 | 6.525204 | 6.525204 | 9.349 | 0.998 | 9.36 |
| 500 | 6.519044 | 6.519044 | 234.484 | 8.170 | 28.70 |
| 1000 | 6.518729 | 6.518729 | 982.074 | 21.084 | 46.58 |

## Development

```bash
make install
make check
make test
make lint
make benchmark
```

## Repository Layout

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── benchmarks/
│   └── benchmark_binomial.py
├── notebooks/
│   └── Binomial_Trees_Homework.ipynb
├── src/
│   └── option_pricing/
│       ├── __init__.py
│       └── binomial.py
├── tests/
│   └── test_binomial.py
├── Makefile
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Limitations

- no dividends,
- constant volatility and interest rate,
- single underlying asset,
- educational/research implementation,
- not a production trading system.
