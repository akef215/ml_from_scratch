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

## Installation

Clone the repository and put it in `regression_lineaire` folder: 

git clone https://github.com/TON_UTILISATEUR/ml-from-scratch.git
cd ml-from-scratch/regression_lineaire

Create and activate a virtual environnement with :

python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate   # Windows

# Dependencies

This project requires the following Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn

Install them using:

pip install -r requirements.txt
pip install -e .

check the installation in Python or the notebook :

from regression_lineaire.src.linear_regression import LinearRegression
