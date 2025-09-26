import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

X, y = datasets.make_classification(n_samples= 1000, n_features= 10, random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
class NaiveBayes:
    def fit(self, X, y):
        n_examples, n_features = X.shape
        self.n_class = np.unique(y)
        len_class = len(self.n_class)
        self._mean = np.zeros((len_class, n_features), dtype= np.float64)
        self._var = np.zeros((len_class, n_features), dtype= np.float64)
        self._prior = np.zeros(len_class, dtype = np.float64)

        for c in self.n_class:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis = 0)
            self._var[c, :] = X_c.var(axis = 0)
            self._prior[c] = X_c.shape[0] / float(n_examples)
    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return prediction
    def _predict(self, x):
        idx_ans = []
        for idx, c in enumerate(self.n_class):
            prior = np.log(self._prior[idx])
            sum_prob = np.sum(np.log(self._calcGaus(idx, x)))
            total = prior + sum_prob
            idx_ans.append(total)
        return self.n_class[np.argmax(idx_ans)]
    def _calcGaus(self, idx, x):
        tu = np.exp(-(x - self._mean[idx]) ** 2 / (2 * self._var[idx]))
        mau = np.sqrt(2 * np.pi * self._var[idx])
        return tu / mau
model = NaiveBayes()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

acc = np.sum(prediction == y_test) / len(y_test) * 100
print(acc)