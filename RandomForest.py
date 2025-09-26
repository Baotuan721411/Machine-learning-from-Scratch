import numpy as np
from Decision_Tree import DecisionTree
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

def split_sample(X, y):
    n_examples = X.shape[0]
    idxs = np.random.choice(n_examples, n_examples, replace = True)
    return X[idxs], y[idxs]
def find_common_label(y):
    most_common = Counter(y).most_common(1)[0][0]
    return most_common
class RandomForest:
    def __init__(self, num_trees = 100, max_depth = 100, min_split_samples = 2, n_features = None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_split_samples = min_split_samples
        self.n_feature = n_features
        self.trees = []
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.num_trees):
            small_tree = DecisionTree(self.max_depth, self.min_split_samples, self.n_feature)
            X_train, y_train = split_sample(X, y)
            small_tree.fit(X_train, y_train)
            self.trees.append(small_tree)
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        predictions = np.swapaxes(predictions, 0, 1)
        most_common_label = [find_common_label(prediction) for prediction in predictions]
        return np.array(most_common_label)

model = RandomForest(num_trees= 3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test) * 100
print(acc)