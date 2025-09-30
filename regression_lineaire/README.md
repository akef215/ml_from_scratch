# LinearRegression from Scratch

This folder contains a **from-scratch implementation of Multiple Linear Regression** using only `numpy`. The class provides two methods for fitting the model:

1. **Normal Equation** (with optional ridge regularization)
2. **Gradient Descent** (with optional input normalization and cost tracking)

---

## Files

- `./linear_regression.py` : Python implementation of the `LinearRegression` class.
- `docs/normal_equation_theory.tex` : LaTeX document explaining the mathematical derivation of the Normal Equation and ridge regularization.
- `docs/gradient_descent_theory.tex` : LaTeX document explaining the mathematical background of the Gradient Descent algorithm.
- `examples/demo.ipynb` : A Jupyter Notebook illustrating the use of the LinearRegression class with visualizations and comparison with scikit-learn.

---

## Dependencies

This project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

Install them using:

```bash
pip install -r requirements.txt
