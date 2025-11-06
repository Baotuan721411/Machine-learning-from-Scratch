import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
np.random.seed(42)
class LogisticRegression:
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
            y_linear = np.dot(X, self.w) + self.b
            y_predict = self.sigmoid(y_linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_samples) * np.sum(y_predict - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        y_linear = np.dot(X, self.w) + self.b
        y_predict = self.sigmoid(y_linear)
        predictions = [1 if i > 0.5 else 0 for i in y_predict]
        return predictions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
'''
model = LogisticRegression(1000, 0.1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test) * 100 
print(acc)
'''
print(bc)
