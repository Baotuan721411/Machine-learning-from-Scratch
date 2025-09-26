from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = datasets.make_regression(n_samples= 500, n_features= 1, noise = 20, random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

class LinearRegression:
    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_examples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            y_predict = np.dot(X, self.weights) + self.bias
            dw = (1 / n_examples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_examples) * np.sum(y_predict - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict
model = LinearRegression(0.05, 1000)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, prediction, c = 'r')
plt.show()
