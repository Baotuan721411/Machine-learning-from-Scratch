from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
iris_data = datasets.load_iris()
X, y = iris_data.data, iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)
def cal_Euclid(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self, X, y):
        self.X = X
        self.y = y
    def predict(self, X):
        labels = [self._predict(x) for x in X]
        return labels
    def _predict(self, x):
        distances = [cal_Euclid(x, x_i) for x_i in self.X]
        k_idx = np.argsort(distances)[:self.k]
        k_label = [self.y[i] for i in k_idx]
        label = Counter(k_label).most_common(1)
        return label[0][0]

model = KNN(3)
model.fit(X_train, y_train)
predicts = model.predict(X_test)
acc = np.sum(predicts == y_test) / len(y_test) * 100
print(acc)