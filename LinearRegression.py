import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from Gradient_Descent import GradientDescent
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
np.random.seed(42)
X, y = datasets.make_regression(n_samples= 500, n_features= 1, noise = 20, random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)
modelLR = LinearRegression()
modelLR.fit(X_train, y_train)
predict = modelLR.predict(X_test)
GD = GradientDescent(X_train, y_train)
class LinearRegression:
    def __init__(self, n_iter, lr):
        self.n_iter = n_iter
        self.lr = lr
        self.w = None
        self.b = None
    def fit(self, X, y):
        self.w, self.b = GD.adam(0.9, 0.99, 1e-8, 2, 2)
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

model = LinearRegression(1000, 0.1)
model.fit(X_train, y_train)
predictionGD = model.predict(X_test)
plt.plot(X_test, predictionGD, c= 'r', linewidth= 10)
modelNE = LinearRegressionNE()
modelNE.fit(X_train, y_train)
predictionsNE = modelNE.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, predictionsNE, c = 'b')
plt.plot(X_test, predict, c = 'g')
plt.show()
mseSK = mean_squared_error(y_test, predict)
mseNE = mean_squared_error(y_test, predictionsNE)
mseGD = mean_squared_error(y_test, predictionGD)
print('Mean square calculated by sklearn: ', mseSK)
print('Mean square calculated by NE: ', mseNE)
print('Mean square calculated by GD: ', mseGD)