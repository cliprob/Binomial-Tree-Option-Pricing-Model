# Binomial Option Pricing Model

Implementation of Cox-Ross-Rubinstein (CRR) binomial tree method for pricing European and American options.

## Features
- European Call/Put options
- American Call/Put options with early exercise
- Efficient NumPy vectorization
- Flexible number of time steps

## Usage
```python
# Example: American Put Option
price, paths, nodes = american_fast_tree(
    K=45, T=1, S0=40, r=0.03, N=100, sigma=0.25, opttype='P'
)
```

## Requirements
- Python 3.x
- NumPy
