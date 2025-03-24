# Regression is predicting continuous value.
# Whekreas in classification we want to predict a discrete value.

# Approximation
# y^hat = wx + b 

# Cost Function : MSE 최대한 작게 fine the minimum 
# 따라서 경사하강법을 계산한다. 
import numpy as np



class LinearRegression:

    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    

    def fit(self, X,y):
        # 시작 파라미터 정하기
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0    

        for _ in range(self.n_iters):
            #추정하기    w = w - alpha * dw
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_sample) * np.dot(X.T , (y_predicted - y))
            db = (1/n_sample) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db





    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    