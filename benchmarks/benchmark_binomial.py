"""Benchmark loop-based and vectorized CRR binomial tree implementations."""

from __future__ import annotations

import time
from collections.abc import Callable

from option_pricing import binomial_tree_price_loop, binomial_tree_price_vectorized

PARAMS = {
    "s0": 40,
    "k": 45,
    "r": 0.03,
    "sigma": 0.25,
    "T": 1,
    "option_type": "put",
    "exercise": "american",
}


def _time_pricer(
    pricer: Callable[..., float],
    n_steps: int,
    repeats: int,
) -> tuple[float, float]:
    start = time.perf_counter()
    price = 0.0
    for _ in range(repeats):
        price = pricer(**PARAMS, n_steps=n_steps)
    elapsed_ms = (time.perf_counter() - start) * 1_000 / repeats
    return price, elapsed_ms


def main() -> None:
    print(
        "| N | Loop price | Vectorized price | Loop time (ms) | "
        "Vectorized time (ms) | Speedup |"
    )
    print("|---:|---:|---:|---:|---:|---:|")

    for n_steps in [100, 500, 1000]:
        repeats = 5
        loop_price, loop_ms = _time_pricer(
            binomial_tree_price_loop,
            n_steps=n_steps,
            repeats=repeats,
        )
        vectorized_price, vectorized_ms = _time_pricer(
            binomial_tree_price_vectorized,
            n_steps=n_steps,
            repeats=repeats,
        )
        speedup = loop_ms / vectorized_ms

        print(
            f"| {n_steps} | {loop_price:.6f} | {vectorized_price:.6f} | "
            f"{loop_ms:.3f} | {vectorized_ms:.3f} | {speedup:.2f} |"
        )


if __name__ == "__main__":
    main()
