import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples= 500, n_features= 2, centers= 2, cluster_std= 1, random_state = 42)
y = np.array([1 if i == 1 else -1 for i in y])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

class SVM:
    def __init__(self, learning_rate = 0.01, lamda = 0.01, n_iter = 1000):
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_examples, n_features = X.shape
        y_label = np.where(y >= 0, 1, -1)
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_y = np.dot(x_i, self.weights) - self.bias
                if linear_y * y[idx] >= 1:
                    dw = 2 * self.lamda * self.weights
                    self.weights -= self.learning_rate * dw
                else:
                    dw = 2 * self.lamda * self.weights - y_label[idx] * x_i
                    db = y_label[idx]
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
    def predict(self, X):
        y_linear = np.dot(X, self.weights) - self.bias
        return np.sign(y_linear)
model = SVM()
model.fit(X_train, y_train)
prediction = model.predict(X_test)
plt.scatter(X[:, 0], X[:, 1], c = y)
min_x = np.amin(X_train[:, 0])
max_x = np.amax(X_train[:, 0])
y1 = (model.bias - model.weights[0] * min_x) / model.weights[1]
y2 = (model.bias - model.weights[0] * max_x) / model.weights[1]
plt.plot([min_x, max_x], [y1, y2])
plt.show()
acc = np.sum(prediction == y_test) / len(y_test) * 100
print(acc)