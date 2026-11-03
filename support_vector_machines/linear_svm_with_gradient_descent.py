import math
import numpy as np

class LinearSVM:
    def __init__(self, lr=0.001, lambda_param = 0.01, n_epochs = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_epochs = n_epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y= y.to_numpy()
        n_samples, n_features = X.shape
        self.w = [0.0] * n_features
        self.b = 0.0
    

        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    self.w = [wj - self.lr * self.lambda_param * wj for wj in self.w]
                else:
                    self.w = [wj - self.lr * (self.lambda_param * wj - y[i] * X[i][j]) for j, wj in enumerate(self.w)] 
                    self.b -= self.lr * (-y[i])

        return self

    def predict(self, X):
            preds = []
            for x in X:
                linear_output = np.dot(x, self.w) + self.b
                pred = 1 if linear_output >= 0 else 0
                preds.append(pred)
            return preds