import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """
    An implementation of Logistic Regression from scratch only using numpy
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

    def _sigmoid(self, z):
        """
        Sigmoid function that maps any real value between 0 and 1
        Args:
            z (float or numpy.ndarray): The input value or array of values
        Returns:
            The sigmoid of z
        """
        return 1 / (1 + np.exp(-1 * z))

    def _cost(self, X, Y, w, b):
        """
        Cost calculates the cross-entropy loss of a given Training
        couple (X, Y)
        Args:
            X (numpy.ndarray): The training set (n_samples, n_features)
            Y (numpy.ndarray): The targets set (n_samples,)
            w (numpy.ndarray): The parameters of Regression vector (n_features,)
            b (float): The bias of the Regression
        Returns:
            The cost function of the model
        """

        # To avoid log(0) which is undefined
        Y_estimated = np.clip(self._sigmoid(X @ w + b), 1e-15, 1 - 1e-15)
        return -1*np.mean(Y * np.log(Y_estimated) + (1 - Y) * np.log(1 - Y_estimated))

    def _gradient_descent(self, X_train, Y_train, learning_rate, max_iter, 
                        normalise = True, epsilon = 1e-5):
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
            Default is e-5

        Returns:
            original_w (numpy.ndarray): The parameters of Regression vector (n_features,)
            original_b (float): The bias of the Regression
            history (list): History of the evaluation of the cost function 
            and parameters w and b in each iteration 
        """

        muX, sigmaX = np.mean(X_train, axis=0), np.std(X_train, axis=0)
        sigmaX[sigmaX == 0] = 1
        # Normalisation (Z-score scaling):
        if normalise:
            X = (X_train - muX) / sigmaX
        else:
            X = X_train
        
        Y = Y_train
        history = []
        w = np.zeros(X_train.shape[1])
        b = 0
        for i in range(max_iter):
            # residual represents the prediction of X
            residual = self._sigmoid(X @ w + b) - Y
            # dW represents the partial derivative of the Cost function
            #  in respect to w
            dW = X.T @ residual / X.shape[0]
            # dB represents the partial derivative of the cost function 
            # in respect to b 
            dB = np.mean(residual)
            w -= learning_rate*dW
            b -= learning_rate*dB

            # ReScaling the coefficients and the bias to get the original values
            original_w = w/sigmaX if normalise else w
            original_b = b - np.dot(original_w, muX) if normalise else b

            # Saving the trace of the cost function and stopping if the cost
            #  is sufficiently small or the cost is getting constant
            cost = self._cost(X_train, Y_train, original_w, original_b)
            history.append((cost, original_w.copy(), original_b))
            if cost < epsilon: break
            if i > 0 and abs(history[-1][0] - history[-2][0]) < epsilon: break
        return original_w, original_b, history
    
    def fit(self, X, Y, learning_rate=0.01, max_iter=1000, method='normal',normalise = True):
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
            normalise (bool, optional): To chose whether we normalise the inputs or not
            Default is True
        
        Returns:
            An object of LogisticRegression so that we can apply the other methodes, mainly predict
        
        """
        self.coef_, self.bias_, self.history_ = self._gradient_descent(X, Y, learning_rate, max_iter, normalise = normalise)
        return self
    
    def predict_proba(self, X):
        """
          Predict the output vector for a given Testing array

        Args:
            X (numpy.ndarray) : Testing array (n_samples, n_features)

        Returns:
            The vector of predictions probabilities
        """
        return self._sigmoid(X @ self.coef_ + self.bias_)
    
    def predict(self, X, threshold=0.5):
        """
          Predict the output vector for a given Testing array


        Args:
            X (numpy.ndarray) : Testing array (n_samples, n_features)
            threshold (float, optional): The threshold to classify the outputs
            Default is 0.5      
        
        Returns:
            The vector of predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def plot_decision_boundary(self, X_train, Y_train):
        """
          Plot the boundary line along with the training data\
            for a 2D logistic regression model.

        Args:
            X_train (numpy.ndarray): Training features (n_samples, 2)
            Y_train (numpy.ndarray): Targets targets (n_samples,)
        """
        if X_train.shape[1] != 2:
                raise ValueError("X_train must have exactly 2 features \
                                 for 2D plotting.")
        plt.figure()
        class_0 = Y_train == 0
        class_1 = Y_train == 1
        plt.scatter(X_train[class_0, 0], X_train[class_0, 1], \
                     color='blue', label='Class 0', alpha=0.5)
        plt.scatter(X_train[class_1, 0], X_train[class_1, 1], \
                    color='red', label='Class 1', alpha=0.5)
        # Create a grid to plot the decision boundary
        plt.plot(X_train[:, 0], -(self.coef_[0] * X_train[:, 0] + self.bias_)\
                  / self.coef_[1], color='green', label='Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title("Logistic Regression Decision Boundary")
        plt.legend()
        plt.show()
    
    def plot_learning_curve(self):
        """
          Plot the cost function history 
          Raises:
            ValueError: If the model has not been trained yet
        """
        if not self.history_:
            raise ValueError("Cost history is only available for models trained")
        
        costs = [entry[0] for entry in self.history_]
        plt.figure()
        plt.plot(range(len(costs)), costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost Function History')
        plt.grid()
        plt.show()
    
    def accuracy(self, Y_true, Y_pred):
        """
          Calculate the accuracy of the model
          Args:
            Y_true (numpy.ndarray): The true labels (n_samples,)
            Y_pred (numpy.ndarray): The predicted labels (n_samples,)
          Returns:
            The accuracy of the model
        """
        return np.mean(Y_true == Y_pred)
    
    def recall(self, Y_true, Y_pred):
        """
          Calculate the recall of the model
          Args:
            Y_true (numpy.ndarray): The true labels (n_samples,)
            Y_pred (numpy.ndarray): The predicted labels (n_samples,)
          Returns:
            The recall of the model
        """
        true_positives = np.sum((Y_true == 1) & (Y_pred == 1))
        false_negatives = np.sum((Y_true == 1) & (Y_pred == 0))
        if true_positives + false_negatives == 0:
            return 0.0
        return true_positives / (true_positives + false_negatives)
    
    def precision(self, Y_true, Y_pred):
        """
          Calculate the precision of the model
          Args:
            Y_true (numpy.ndarray): The true labels (n_samples,)
            Y_pred (numpy.ndarray): The predicted labels (n_samples,)
          Returns:
            The precision of the model
        """
        true_positives = np.sum((Y_true == 1) & (Y_pred == 1))
        false_positives = np.sum((Y_true == 0) & (Y_pred == 1))
        if true_positives + false_positives == 0:
            return 0.0
        return true_positives / (true_positives + false_positives)

    def get_history(self):
        """
          The getter method of the history of the parameters of the model
          and the cost function at each iteration
          Returns:
            A list of (cost, coef, bias) if self.fit called before. Null otherwise
        """
        return self.history_
    
    def get_params(self):
        """
          The getter method of the parameters of the model
          Returns:
            A tuple (coef, bias)
        """
        return self.coef_, self.bias_
    
