import math

import pytest

from option_pricing import (
    binomial_tree_price_loop,
    binomial_tree_price_vectorized,
    black_scholes_price,
)

BASE_PARAMS = {
    "s0": 40,
    "k": 45,
    "r": 0.03,
    "sigma": 0.25,
    "T": 1,
}


@pytest.mark.parametrize(
    ("option_type", "exercise"),
    [
        ("call", "european"),
        ("put", "european"),
        ("call", "american"),
        ("put", "american"),
    ],
)
def test_vectorized_and_loop_implementations_agree(option_type, exercise):
    params = BASE_PARAMS | {
        "n_steps": 100,
        "option_type": option_type,
        "exercise": exercise,
    }

    vectorized_price = binomial_tree_price_vectorized(**params)
    loop_price = binomial_tree_price_loop(**params)

    assert vectorized_price == pytest.approx(loop_price, abs=1e-10)


@pytest.mark.parametrize("n_steps", [50, 100, 250])
def test_european_call_converges_to_black_scholes(n_steps):
    binomial_price = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=n_steps,
        option_type="call",
        exercise="european",
    )
    black_scholes = black_scholes_price(**BASE_PARAMS, option_type="call")

    assert binomial_price == pytest.approx(black_scholes, abs=5e-2)


@pytest.mark.parametrize("n_steps", [50, 100, 250])
def test_european_put_converges_to_black_scholes(n_steps):
    binomial_price = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=n_steps,
        option_type="put",
        exercise="european",
    )
    black_scholes = black_scholes_price(**BASE_PARAMS, option_type="put")

    assert binomial_price == pytest.approx(black_scholes, abs=5e-2)


def test_put_call_parity_approximately_holds_for_european_options():
    call_price = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="call",
        exercise="european",
    )
    put_price = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="put",
        exercise="european",
    )

    parity_value = BASE_PARAMS["s0"] - BASE_PARAMS["k"] * math.exp(
        -BASE_PARAMS["r"] * BASE_PARAMS["T"]
    )

    assert call_price - put_price == pytest.approx(parity_value, abs=1e-10)


def test_american_put_is_worth_at_least_european_put():
    european_put = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="put",
        exercise="european",
    )
    american_put = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="put",
        exercise="american",
    )

    assert american_put >= european_put


def test_american_call_without_dividends_matches_european_call():
    european_call = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="call",
        exercise="european",
    )
    american_call = binomial_tree_price_vectorized(
        **BASE_PARAMS,
        n_steps=250,
        option_type="call",
        exercise="american",
    )

    assert american_call == pytest.approx(european_call, abs=1e-8)


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"s0": -40},
        {"k": 0},
        {"sigma": 0},
        {"T": 0},
        {"option_type": "straddle"},
        {"exercise": "bermudan"},
        {"n_steps": 0},
    ],
)
def test_invalid_inputs_raise_value_error(invalid_kwargs):
    params = BASE_PARAMS | {
        "n_steps": 100,
        "option_type": "call",
        "exercise": "european",
    }
    params.update(invalid_kwargs)

    with pytest.raises(ValueError):
        binomial_tree_price_vectorized(**params)
