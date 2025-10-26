import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.random.seed()
X, y = datasets.make_regression(n_samples= 500, n_features= 1, noise = 20, random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)
class LinearRegressionGD:
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.w = None
        self.b = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = np.random.randn() * 0.01
        for _ in range(self.n_iter):
            y_predict = np.dot(X, self.w) + self.b
            dw = 1 / n_samples * np.dot(X.T, (y_predict - y))
            db = 1 / n_samples * np.sum(y_predict - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        predictions = np.dot(X, self.w) + self.b
        return predictions
class LinearRegressionNE:
    def __init__(self):
        self.w = None
        self.X = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        ones = np.ones((n_samples, 1))
        self.X = np.concatenate((ones, X), axis= 1)
        X_multiply = np.dot(self.X.T, self.X)
        self.w = np.dot(np.dot(np.linalg.pinv(X_multiply), self.X.T), y)
    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_new = np.concatenate((ones, X), axis= 1)
        predictions = np.dot(X_new, self.w)
        return predictions
class LinearRegressionSGD:
    def __init__(self, n_iter, lr):
        self.w = None
        self.b = None
        self.n_iter = n_iter
        self.lr = lr
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.random.randn(n_features) * 0.01
        self.b = np.random.randn() * 0.01

        for _ in range(self.n_iter):
            for i in range(n_samples):
                y_predict = np.dot(X[i], self.w) + self.b
                dw = X[i] * (y_predict - y[i])
                db = y_predict - y[i]

                self.w -= self.lr * dw
                self.b -= self.lr * db
    def predict(self, X):
        predictions = np.dot(X, self.w) + self.b
        return predictions

modelGD = LinearRegressionGD(1000, 0.1)
modelGD.fit(X_train, y_train)
predictionsGD = modelGD.predict(X_test)
plt.plot(X_test, predictionsGD, c= 'r', linewidth= 10)
modelNE = LinearRegressionNE()
modelNE.fit(X_train, y_train)
predictionsNE = modelNE.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, predictionsNE, c = 'b')
modelSGD = LinearRegressionSGD(1000, 0.1)
modelSGD.fit(X_train, y_train)
predictionsSGD = modelSGD.predict(X_test)
plt.plot(X_test, predictionsSGD, c= 'y', linewidth= 5)
plt.show()