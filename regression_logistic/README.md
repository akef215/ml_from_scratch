# LogisticRegression from Scratch

This folder contains a **from-scratch implementation of Logistic Regression** using only `numpy`. The class uses **Gradient Descent** (with optional input normalization and cost tracking) for fitting the model:

---

## Files

- `./logistic_regression.py` : Python implementation of the `LogisticRegression` class.
- `examples/demo.ipynb` : A Jupyter Notebook illustrating the use of the LogisticRegression class with visualizations and comparison with scikit-learn.

---

## Installation

Clone the repository and put it in `regression_logistic` folder: 

git clone https://github.com/TON_UTILISATEUR/ml-from-scratch.git
cd ml-from-scratch/regression_logistic

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

from regression_logistic.src.logistic_regression import LogisticRegression
