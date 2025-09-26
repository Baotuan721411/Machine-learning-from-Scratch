import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(42)
X, y = datasets.make_blobs(n_samples= 5000, n_features = 2, centers = 3, random_state= 42)

plt.scatter(X[:, 0], X[:, 1], c = y)
def cal_Euclid(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class K_means_clustering:
    def __init__(self, K = 3, n_iters = 10000):
        self.K = K
        self.n_iters = n_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centrals = []
    def predict(self, X):
        self.X = X
        self.examples, self.features = X.shape
        # tạo central ngẫu nhiên
        random_idx = np.random.choice(self.examples, size= self.K, replace = False)
        self.centrals = [self.X[idx] for idx in random_idx]
        for _ in range(self.n_iters):
            # phân cụm cho central
            self.clusters = self._make_clusters(self.centrals)
            # lưu central cũ
            old_central = self.centrals
            # cập nhật central mới
            self.centrals = self._update_central(self.clusters)
            # kiểm tra hội tụ
            if self.converge(old_central, self.centrals):
                break
        return self.get_cluster_label(self.clusters)

    def get_cluster_label(self, clusters):
        label = np.empty(self.examples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in  cluster:
                label[sample_idx] = cluster_idx
        return label
    def converge(self, old, new):
        distances = [cal_Euclid(old[i], new[i]) for i in range(self.K)]
        return sum(distances) == 0
    def _make_clusters(self, centrals):
        cluster = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            closet_central_idx = self._find_closet_central_idx(sample, centrals)
            cluster[closet_central_idx].append(idx)
        return cluster

    def _find_closet_central_idx(self, sample, centrals):
        distance = [cal_Euclid(sample, central) for central in centrals]
        return np.argmin(distance)
    def _update_central(self, clusters):
        central = np.zeros((self.K, self.features))
        for idx, cluster in enumerate(clusters):
            central[idx] = np.mean(self.X[cluster], axis = 0)
        return central

model = K_means_clustering(3)
prediction = model.predict(X)
for i in range(model.K):
    plt.scatter(model.centrals[i][0], model.centrals[i][1], marker= 'x')
plt.show()