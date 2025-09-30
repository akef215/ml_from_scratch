import numpy as np

class KNN:
    """
    An implementation of K Nearest Neighbours Algorithm from scratch only using numpy
    Constructor
    Attributs : 
        coef_ (numpy.ndarray): coefficients of Regression (numpy array (m,))
        bias_ (float): bias of Regression 
        history_ (list): the history of the costs in each iteration if
        you use the Gradient Descent method otherwise it returns []
    """
    def __init__(self):
        self.coef_ = None
        self.bias_ = 0

    def _cost(self, X, Y, w, b):
        """
        Cost calculates the Mean Squared Error (MSE) of a given Training couple (X, Y)
        Args:
            X (numpy.ndarray): The training set (n_samples, n_features)
            Y (numpy.ndarray): The targets set (n_samples,)
            w (numpy.ndarray): The parameters of Regression vector (n_features,)
            b (float): The bias of the Regression
        Returns:
            The cost function of the model
        """

        return np.mean((X @ w + b - Y)**2)/2

    def fit(self, X, Y, ridge_coefficient = 1e-2):
        """
          Train the model on the data X (Observations) 
          and Y (Labels)

        Args:
            X (numpy.ndarray): Training features (n_samples, n_features)
            Y (numpy.ndarray): Targets targets (n_samples,)
            learning_rate (float, optional) : The step size for gradient update
            Default is 0.01
            max_iter (int, optional): The maximum number of iterations
            Default is 1000
            method (str, optional): the method using for regression it can be
            'normal' for Normale Equation method or 'gradient' for 
            Gradient Descent. Default is 'normal'
            ridge_coefficient (float, optional): a float coefficient to ensure
            that the normal equation admits a solution
            Default is 1e-2
            normalise (bool, optional): To chose whether we normalise the inputs or not
            Default is True
        
        Returns:
            An object of LinearRegression so that we can apply the other methodes, mainly predict
        
        
        Notes
        -----
        For the mathematical derivation of the Normal Equation and 
        its ridge regularization extension, please refer to the 
        accompanying LaTeX document: `normal_equation_theory.pdf` in the docs folder.

        """
        # The ridge coefficient to ensure that the normal equation 
        # admits a solution
        # Add a ones column to calulate the bias
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        # Normal Formula with ridge
        D = np.eye(X.shape[1])
        # We don't ridge the bias
        D[0, 0] = 0
        # The solution of the normal equation
        parameters = np.linalg.solve(X.T @ X + ridge_coefficient * D, X.T @ Y)
        self.coef_ = parameters[1:]
        self.bias_ = parameters[0]
        return self
    
    def predict(self, X):
        """
          Predict the output vector for a given Testing array

        Args:
            X (numpy.ndarray) : Testing array (n_samples, n_features)

        Returns:
            The vector of predictions
        """
        return X @ self.coef_ + self.bias_
    