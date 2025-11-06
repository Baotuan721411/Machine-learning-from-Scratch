import numpy as np

class GradientDescent:
    def __init__(self, X, y, n_iter = 1000, lr = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape

        self.w, self.b = None, None
    def _init_params(self):
        self.w = np.random.randn(self.n_features) * 0.01
        self.b = np.random.randn() * 0.01
    def GD(self): # với mỗi iter lại hải tính lại tích dot của ma trận * vector => tốn time
        self._init_params()
        for _ in range(self.n_iter):
            y_linear = np.dot(self.X, self.w) + self.b
            dw = (1 / self.n_samples) * np.dot(self.X.T, (y_linear - self.y))
            db = (1 / self.n_samples) * np.sum(y_linear - self.y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self.w, self.b
    def SGD(self): # đã khắc phục khi chỉ tính với 1 sample, nhưng vẫn có thể bị mắc kẹt ở local minimal
        self._init_params()
        for _ in range(self.n_iter):
            for idx in range(self.n_samples):
                y_linear = np.dot(self.X[idx], self.w) + self.b
                dw = self.X[idx] * (y_linear - self.y[idx])
                db = y_linear - self.y[idx]
                self.w -= self.lr * dw
                self.b -= self.lr * db
        return self.w, self.b
    def HeavyBall(self, beta): # vẫn khó hội tụ khi chọn hệ số beta không phù hợp
        self._init_params()
        self.previous_w, self.previous_b = np.zeros_like(self.w), 0
        for _ in range(self.n_iter):
            for idx in range(self.n_samples):
                y_linear = np.dot(self.X[idx], self.w) + self.b
                dw = self.X[idx] * (y_linear - self.y[idx])
                db = y_linear - self.y[idx]
                cur_w, cur_b = self.w, self.b
                self.w = self.w - self.lr * dw + beta * (self.w - self.previous_w)
                self.b = self.b - self.lr * db + beta * (self.b - self.previous_b)
                self.previous_w, self.previous_b = cur_w, cur_b
        return self.w, self.b
    def NAG(self, beta): # Nesterov Accelarated Gradient, nhìn trước để đi đúng hơn, hội tụ nhanh hơn, nhưng vẫn khó với việc chọn hệ số beta
        self._init_params()
        v_w = np.zeros_like(self.w)
        v_b = 0

        for _ in range(self.n_iter):
            idxs = np.arange(self.n_samples)
            np.random.shuffle(idxs)
            for idx in idxs:
                w_new = self.w - beta * v_w
                b_new = self.b - beta * v_b

                y_linear = np.dot(self.X[idx], w_new) + b_new
                error = y_linear - self.y[idx]
                dw = np.dot(self.X[idx], error)
                db = np.sum(error)

                v_w  = beta * v_w + self.lr * dw
                v_b = beta * v_b + self.lr * db
                self.w -= v_w
                self.b -= v_b
        return self.w, self.b


