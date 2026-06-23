"""CRR binomial tree pricing for European and American options."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

VALID_OPTION_TYPES = {"call", "put"}
VALID_EXERCISE_TYPES = {"european", "american"}


def _validate_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _validate_n_steps(n_steps: int) -> None:
    if isinstance(n_steps, bool) or not isinstance(n_steps, Integral) or n_steps < 1:
        raise ValueError("n_steps must be an integer greater than or equal to 1")


def _validate_market_inputs(
    s0: float | None,
    k: float | None,
    r: float,
    sigma: float,
    T: float,
    n_steps: int | None = None,
) -> None:
    if s0 is not None:
        _validate_positive("s0", s0)
    if k is not None:
        _validate_positive("k", k)
    if not math.isfinite(r):
        raise ValueError("r must be a finite number")
    _validate_positive("sigma", sigma)
    _validate_positive("T", T)
    if n_steps is not None:
        _validate_n_steps(n_steps)


def _validate_option_type(option_type: str) -> str:
    if option_type not in VALID_OPTION_TYPES:
        raise ValueError("option_type must be either 'call' or 'put'")
    return option_type


def _validate_exercise_type(exercise: str) -> str:
    if exercise not in VALID_EXERCISE_TYPES:
        raise ValueError("exercise must be either 'european' or 'american'")
    return exercise


def _payoff(stock_prices: np.ndarray | float, k: float, option_type: str) -> np.ndarray:
    if option_type == "call":
        return np.maximum(np.asarray(stock_prices) - k, 0.0)
    return np.maximum(k - np.asarray(stock_prices), 0.0)


def crr_parameters(
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
) -> tuple[float, float, float]:
    """Return CRR up factor, down factor, and risk-neutral probability."""
    _validate_market_inputs(
        s0=None,
        k=None,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
    )

    dt = T / n_steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)

    if p < 0.0 or p > 1.0:
        raise ValueError("risk-neutral probability must be in [0, 1]")

    return float(u), float(d), float(p)


def binomial_tree_price_vectorized(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """Price a European or American option using a vectorized CRR binomial tree."""
    _validate_market_inputs(s0=s0, k=k, r=r, sigma=sigma, T=T, n_steps=n_steps)
    option_type = _validate_option_type(option_type)
    exercise = _validate_exercise_type(exercise)

    u, d, p = crr_parameters(r=r, sigma=sigma, T=T, n_steps=n_steps)
    dt = T / n_steps
    discount = math.exp(-r * dt)

    node_indices = np.arange(n_steps + 1)
    stock_prices = s0 * (u ** (n_steps - node_indices)) * (d**node_indices)
    option_values = _payoff(stock_prices, k, option_type)

    for step in range(n_steps - 1, -1, -1):
        option_values = discount * (
            p * option_values[:-1] + (1.0 - p) * option_values[1:]
        )

        if exercise == "american":
            step_indices = np.arange(step + 1)
            step_prices = s0 * (u ** (step - step_indices)) * (d**step_indices)
            option_values = np.maximum(
                option_values,
                _payoff(step_prices, k, option_type),
            )

    return float(option_values[0])


def binomial_tree_price_loop(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    option_type: str = "call",
    exercise: str = "european",
) -> float:
    """Reference loop-based CRR implementation for validation and benchmarking."""
    _validate_market_inputs(s0=s0, k=k, r=r, sigma=sigma, T=T, n_steps=n_steps)
    option_type = _validate_option_type(option_type)
    exercise = _validate_exercise_type(exercise)

    u, d, p = crr_parameters(r=r, sigma=sigma, T=T, n_steps=n_steps)
    dt = T / n_steps
    discount = math.exp(-r * dt)

    option_values = []
    for node in range(n_steps + 1):
        stock_price = s0 * (u ** (n_steps - node)) * (d**node)
        option_values.append(float(_payoff(stock_price, k, option_type)))

    for step in range(n_steps - 1, -1, -1):
        next_values = []
        for node in range(step + 1):
            continuation_value = discount * (
                p * option_values[node] + (1.0 - p) * option_values[node + 1]
            )

            if exercise == "american":
                stock_price = s0 * (u ** (step - node)) * (d**node)
                exercise_value = float(_payoff(stock_price, k, option_type))
                continuation_value = max(continuation_value, exercise_value)

            next_values.append(continuation_value)
        option_values = next_values

    return float(option_values[0])


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    s0: float,
    k: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = "call",
) -> float:
    """Black-Scholes closed-form price for European call/put options."""
    _validate_market_inputs(s0=s0, k=k, r=r, sigma=sigma, T=T)
    option_type = _validate_option_type(option_type)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    discounted_strike = k * math.exp(-r * T)

    if option_type == "call":
        price = s0 * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    else:
        price = discounted_strike * _normal_cdf(-d2) - s0 * _normal_cdf(-d1)

    return float(price)
