import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

data = datasets.load_iris()
X = data.data
y = data.target

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.component = None
        self.mean = None
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X_center = X - self.mean

        cov = np.cov(X_center.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idxs = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, idxs]

        self.component = eigenvectors[:, 0 : self.n_components]
    def transform(self, X):
        X_center = X - self.mean
        return np.dot(X_center, self.component)

model = PCA(2)
model.fit(X)
X_new = model.transform(X)
x1 = X_new[:, 0]
x2 = X_new[:, 1]
plt.scatter(x1, x2, c = y)
plt.show()