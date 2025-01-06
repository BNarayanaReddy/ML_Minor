# Multiple Linear Regression
# function => f(x) = w transposed * x + b
# Loss => Mean Squared Error loss
# Gradientdescent => w = w - lr * gradient

import numpy as np
class LinearRegression:
    def __init__(self, lr=0.001, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        
    def initialize_weights(self, X):
        m, n = X.shape
        self.W = np.zeros(n)
        self.b = 0
    
    def predict(self, X):
        return np.matmul(X, self.W) + self.b
    
    def cost(self, X, y):
        # average losses = cost
        m = len(X)
        predictions = self.predict(X)
        return np.sum((predictions - y)**2)/m
    
    def gradient(self, X, y):
        m = len(X)
        predictions = self.predict(X)
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        return dw, db
    
    def gradient_descent(self, X, y):
        m = len(X)
        dw, db = self.gradient(X, y)
        self.W = self.W - self.lr * dw
        self.b = self.b - self.lr * db

        parameters = {'W': self.W, 'b': self.b}
        return parameters

    def fit(self, X, y):
        m, n = X.shape
        self.initialize_weights(X)
        # gradient descent
        for i in range(self.epochs):
            parameters = self.gradient_descent(X, y)
            if i % 100 == 0:
                print(f"Epoch {i+1}: Cost = {self.cost(X, y)}")
        return parameters

# Use case

# model = LinearRegression(lr=0.01, epochs=1000)
# X = np.array([[1, 2, 4], [3, 4, 5], [5, 6, 7]])
# y = np.array([1, 2, 3])
# model.fit(X, y)
# print(model.W, model.b)
# # test model
# test_x = model.predict(np.array([[1, 2, 4]]))
# print(test_x)
# print(f'train error : ', test_x - y[0]) # train error : 0.0101
