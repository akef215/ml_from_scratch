# LinearRegression from Scratch

This folder contains a **from-scratch implementation of Multiple Linear Regression** using only `numpy`. The class provides two methods for fitting the model:

1. **Normal Equation** (with optional ridge regularization)
2. **Gradient Descent** (with optional input normalization and cost tracking)

---

## Files

- `src/linear_regression.py` : Python implementation of the `LinearRegression` class.
- `docs/normal_equation_theory.pdf` : LaTeX document explaining the mathematical derivation of the Normal Equation and ridge regularization.
- `docs/gradient_descent_theory.pdf` : LaTeX document explaining the mathematical background of the Gradient Descent algorithm.
- (optional) `examples/demo.ipynb` : A Jupyter Notebook illustrating the use of the LinearRegression class with visualizations and comparison with scikit-learn.

---

## Installation

This package requires the libraries listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
