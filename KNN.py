import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = datasets.load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
print(X.shape)
plt.scatter(X_train[:, 0], X_train[:, 1], c = y_train)
plt.show()
def cal_euclid(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
class KNN:
    def __init__(self, K = 3):
        self.k = K
    def fit(self, X, y):
        self.X = X
        self.y = y
    def predict(self, X):
        predictions = [self._predict(x) for x in  X]
        return predictions
    def _predict(self, x):
        distances = [cal_euclid(x_i, x) for x_i in self.X]
        k_idx = np.argsort(distances)[:self.k]
        k_label = [self.y[i] for i in k_idx]
        most_common = Counter(k_label).most_common(1)
        return most_common[0][0]

model = KNN()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test) * 100
print(acc)