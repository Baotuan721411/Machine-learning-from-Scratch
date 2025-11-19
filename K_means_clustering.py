import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples= 5000, n_features= 2, centers= 3, random_state= 42)
plt.scatter(X[:, 0], X[:, 1], c = y)
def cal_Euclid(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class K_means_clustering:
    def __init__(self, n_iter = 1000, K = 3):
        self.K = K
        self.n_iter = n_iter
        self.centers = []
        self.clusters = [[] for i in range(K)]
        self.X = None
        self.n_samples, self.n_features = None, None
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        random_idx = np.random.choice(self.n_samples, size = self.K, replace = False)
        self.centers = [self.X[idx] for idx in random_idx]
        for i in range(self.n_iter):
            self.clusters = self.make_cluster(self.centers)
            old_centers = self.centers
            self.centers = self.make_new_centers(self.clusters)
            if self.converge(old_centers, self.centers):
                break
        return self.get_cluster_label(self.clusters)
    def get_cluster_label(self, clusters):
        label = np.empty(self.n_samples)
        for center_idx, cluster in enumerate(clusters):
            for cluster_idx in cluster:
                label[cluster_idx] = center_idx
        return label
    def make_cluster(self, centers):
        clusters = [[] for i in range(self.K)]
        for idx, point in enumerate(self.X):
            distance = []
            for i in range(self.K):
                dis_to_center_ith = cal_Euclid(point, centers[i])
                distance.append(dis_to_center_ith)
            idx_min = np.argmin(distance)
            clusters[idx_min].append(idx)
        return clusters
    def make_new_centers(self, clusters):
        centers = np.zeros((self.K, self.n_features))
        for idx, cluster in enumerate(clusters):
            centers[idx] = np.mean(self.X[cluster], axis = 0)
        return centers
    def converge(self, old, new):
        distance = [cal_Euclid(old[idx], new[idx]) for idx in range(self.K)]
        return np.sum(distance) == 0

model = K_means_clustering()
prediction = model.predict(X)
for x, y in model.centers:
    plt.scatter(x, y, marker= 'x', c = 'k')
plt.show()
