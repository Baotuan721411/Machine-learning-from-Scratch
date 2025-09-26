import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples= 500, n_features= 2, centers = 2, cluster_std= 1.0, random_state = 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

class Perceptron:
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        self.lr = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.funct = self.func
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                y_linear = np.dot(x_i, self.weights) + self.bias
                y_predict = self.funct(y_linear)
                update = (y[idx] - y_predict) * self.lr
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        y_linear = np.dot(X, self.weights) + self.bias
        predictions = self.funct(y_linear)
        return predictions
    def func(self, x):
        return np.where(x >= 0, 1, 0)
model = Perceptron()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train)
x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])
x1_1 = (-model.bias - model.weights[0] * x0_1) / model.weights[1]
x1_2 = (-model.bias - model.weights[0] * x0_2) / model.weights[1]
plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.show()
acc = np.sum(predictions == y_test) / len(y_test) * 100
print(acc)
print(model.weights, model.bias)
# commit để sai doạn code 
