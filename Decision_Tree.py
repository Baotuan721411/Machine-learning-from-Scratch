import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

data = datasets.load_breast_cancer()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
def entropy(y):
    cnt = np.bincount(y)
    ps = cnt / len(y)
    return - np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, features = None, threshold = None, left = None, right = None, *, val = None):
        self.features = features
        self.threshold = threshold
        self.left = left
        self.right = right
        self.val = val
    def _is_leaf_node(self):
        return self.val is not None

class DecisionTree:
    def __init__(self, max_depth = 100, min_sample_split = 2, features = None):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.features = features
        self.root = None
    def fit(self, X, y):
        self.features = X.shape[1] if self.features is None else min(self.features, X.shape[1])
        self.root = self.make_tree(X, y)

    def make_tree(self, X, y, depth = 0):
        n_examples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_examples < self.min_sample_split:
            common_label = self.find_common(y)
            return Node(val = common_label)

        features_idxs = np.random.choice(n_features, self.features, replace = False)
        best_feature, best_threshold = self.find_best_criteria(X, y, features_idxs)
        left_idx, right_idx = self.split(X[:, best_feature], best_threshold)
        left = self.make_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self.make_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feature, best_threshold, left, right)
    def find_best_criteria(self, X, y, features_idxs):
        best_ig = -1
        best_feature, best_threshold = None, None
        for idx in features_idxs:
            X_col = X[:, idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                ig = self.calc_gain(y, X_col, threshold)
                if ig > best_ig:
                    best_ig = ig
                    best_feature = idx
                    best_threshold = threshold
        return best_feature, best_threshold
    def calc_gain(self, y, X_col, threshold):
        # cal parent entropy
        parent_entropy = entropy(y)
        # split
        left_idx, right_idx = self.split(X_col, threshold)
        n_l, n_r = len(left_idx), len(right_idx)
        n = len(X_col)
        e_l = entropy(y[left_idx])
        e_r = entropy(y[right_idx])
        children = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - children
    def split(self, X_col, threshold):
        left_idx = np.argwhere(X_col <= threshold).flatten()
        right_idx = np.argwhere(X_col > threshold).flatten()
        return left_idx, right_idx
    def find_common(self, y):
        most_common = Counter(y).most_common(1)[0][0]
        return most_common
    def predict(self, X):
        return np.array([self.traverse(x, self.root) for x in X])
    def traverse(self, x, Node):
        if Node._is_leaf_node():
            return Node.val
        if x[Node.features] <= Node.threshold:
            return self.traverse(x, Node.left)
        return self.traverse(x, Node.right)

if __name__ == "__main__":
    model = DecisionTree()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test) * 100
    print(acc)
