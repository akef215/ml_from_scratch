import numpy as np

class LinearRegression:
    """
    An implementation of Multiple Linear Regression from scratch only using numpy
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
        self.history_ = []

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

    def _gradient_descent(self, X_train, Y_train, learning_rate, max_iter, 
                        normalise = True, epsilon = 1e-8):
        """
        It calculates the local minima of a given function : X_train -> Y_train
        Args:
            X_train (numpy.ndarray): Training features (n_samples, n_features)
            Y_train (numpy.ndarray): Targets targets (n_samples,)
            learning_rate (float) : The step size for gradient update
            max_iter (int): The maximum number of iterations
            normalise (bool, optional): To chose whether we normalise the inputs or not
            Default is True
            epsilon (float, optional) : The tolerated error or dCost to stop the execution.
            Default is e-8

        Returns:
            original_w (numpy.ndarray): The parameters of Regression vector (n_features,)
            original_b (float): The bias of the Regression
            history (list): History of the evaluation of the cost function 
            and parameters w and b in each iteration 
        """

        muX, sigmaX = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        muY, sigmaY = np.mean(Y_train), np.std(Y_train)
        sigmaX[sigmaX == 0] = 1
        # Normalisation (Z-score scaling):
        if normalise:
            X = (X_train - muX) / sigmaX
            Y = (Y_train - muY) / sigmaY
        else:
            X, Y = X_train, Y_train
    
        history = []
        w = np.zeros(X_train.shape[1])
        b = 0
        for i in range(max_iter):
            # residual represents the prediction of X
            residual = X @ w + b - Y
            # dW represents the partial derivative of the Cost function in respect to w
            dW = X.T @ residual / X.shape[0]
            #dB represents the partial derivative of the cost function in respect to b 
            dB = np.mean(residual)
            w -= learning_rate*dW
            b -= learning_rate*dB

            # ReScaling the coef
            original_w = w*sigmaY/sigmaX
            original_b = b*sigmaY + muY - np.dot(original_w, muX)

            # Saving the trace of the cost function and stopping if the cost
            #  is sufficiently small or the cost is getting constant
            cost = self._cost(X_train, Y_train, original_w, original_b)
            history.append((cost, original_w.copy(), original_b))
            if cost < epsilon: break
            if i > 0 and abs(history[-1][0] - history[-2][0]) < epsilon: break
        return original_w, original_b, history
    
    def fit(self, X, Y, learning_rate=0.01, max_iter=1000, method='normal', ridge_coefficient = 1e-2,normalise = True):
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
        accompanying LaTeX document: `normal_equation_theory.pdf`.

        """
        if method == 'normal':
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

        elif method == 'gradient':
            self.coef_, self.bias_, self.history_ = self._gradient_descent(X, Y, learning_rate, max_iter, normalise = normalise)
        else:
            raise ValueError("The method must be normal or gradient")
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
    
    def get_history(self):
        """
          The getter method of the history of the parameters of the model
          and the cost function at each iteration
          Returns:
            A list of (cost, coef, bias) if self.fit called before. Null otherwise
        """
        return self.history_