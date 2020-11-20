import numpy as np

class LinearRegre:
    def __init__(self, a, n_iters):
        self.a = a
        self.n_iters = n_iters
        self.weights = None
        self.bias = None 

    def fit(self, x, y):   
        # init parameters
        self.n_samples, self.n_features = x.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.n_iters): #gradient descent to find weight and bias
            self.y_predicted = np.dot(x, self.weights) + self.bias   # y = wx + b, y prediction from current w, x, b 

            self.dw = (1/self.n_samples) + 2 * (np.dot(x.T,(self.y_predicted - y))) # be carefule which way round  when using np.dot to do maxtric/vector multiplication  
            self.db = (1/self.n_samples) + 2 * (np.sum((self.y_predicted - y)))

            self.weights = self.weights - self.a * self.dw
            self.bias = self.bias - self.a * self.db

    def predict(self, x): 
        self.y = np.dot(x, self.weights) + self.bias 
        return self.y