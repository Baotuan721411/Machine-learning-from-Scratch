import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
class LogisticRegression:
    def __init__(self, learning_rate, n_iter):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weight = None
        self.bias = None
    def fit(self, X, y):
        n_examples, n_features = X.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            cal_y = np.dot(X, self.weight) + self.bias
            y_predict = self.cal_sigmoid(cal_y)

            dw = (1 / n_examples) * np.dot(X.T, (y_predict - y))
            db = (1 / n_examples) * np.sum(y_predict - y)

            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        cal_y = np.dot(X, self.weight) + self.bias
        y_predict = self.cal_sigmoid(cal_y)
        predicions = [1 if i > 0.5 else 0 for i in y_predict]
        return predicions
    def cal_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
model = LogisticRegression(0.01, 1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test) * 100
print(acc)