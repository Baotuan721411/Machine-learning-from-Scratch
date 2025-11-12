import numpy as np

class GradientDescent:
    def __init__(self, X, y, n_iter = 1000, lr = 0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.X, self.y = X, y
        self.n_samples, self.n_features = X.shape
        self.w, self.b = None, None
        self.lamda = 0.01
    def _init_params(self):
        self.w = np.random.randn(self.n_features) * 0.01
        self.b = np.random.randn() * 0.01
    def calc_derivative(self, X, y, w, b, regu = None):
        y_linear = np.dot(X, w) + b
        error = y_linear - y
        dw = (1 / len(X)) * np.dot(X.T, error)
        db = (1 / len(X)) * np.sum(error)

        if regu == 1:
            dw += self.lamda * np.sum(np.sign(w))
            db += self.lamda * np.sign(b)
        elif regu == 2:
            dw += 2 * self.lamda * np.sum(w)
            db += 2 * self.lamda * b
        return dw, db
    def GD(self, regu = None): # với mỗi iter lại hải tính lại tích dot của ma trận * vector => tốn time
        self._init_params()
        for _ in range(self.n_iter):
            dw, db = self.calc_derivative(self.X, self.y, self.w, self.b, regu)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self.w, self.b
    def SGD(self, regu = None): # đã khắc phục khi chỉ tính với 1 sample, nhưng vẫn có thể bị mắc kẹt ở local minimal
        self._init_params()
        for _ in range(self.n_iter):
            for idx in range(self.n_samples):
                dw, db = self.calc_derivative(self.X[idx], self.y[idx], self.w, self.b, regu)
                self.w -= self.lr * dw
                self.b -= self.lr * db
        return self.w, self.b
    def HeavyBall(self, beta = 0.01, regu = None): # vẫn khó hội tụ khi chọn hệ số beta không phù hợp
        self._init_params()
        self.previous_w, self.previous_b = np.zeros_like(self.w), 0
        for _ in range(self.n_iter):
            for idx in range(self.n_samples):
                dw, db = self.calc_derivative(self.X[idx], self.y[idx], self.w, self.b, regu)
                cur_w, cur_b = self.w, self.b
                self.w = self.w - self.lr * dw + beta * (self.w - self.previous_w)
                self.b = self.b - self.lr * db + beta * (self.b - self.previous_b)
                self.previous_w, self.previous_b = cur_w, cur_b
        return self.w, self.b
    def NAG(self, beta = 0.01, regu = None): # Nesterov Accelarated Gradient, nhìn trước để đi đúng hơn, hội tụ nhanh hơn, nhưng vẫn khó với việc chọn hệ số beta
        self._init_params()
        v_w = np.zeros_like(self.w)
        v_b = 0

        for _ in range(self.n_iter):
            idxs = np.arange(self.n_samples)
            np.random.shuffle(idxs)
            for idx in idxs:
                w_new = self.w - beta * v_w
                b_new = self.b - beta * v_b

                dw, db = self.calc_derivative(self.X[idx], self.y[idx], w_new, b_new, regu)

                v_w  = beta * v_w + self.lr * dw
                v_b = beta * v_b + self.lr * db
                self.w -= v_w
                self.b -= v_b
        return self.w, self.b
    def mini_batch(self, regu = None):
        self._init_params()
        batch_size = 32  #Số sample cần dùng cho 1 lần cập nhập
        epoch = 100 # số lần lặp qua toàn bộ dataset
        for _ in range(epoch):
            shuffle_index = np.random.permutation(len(self.X))
            X_shuffle = self.X[shuffle_index]
            y_shuffle = self.y[shuffle_index]
            for i in range(0, len(self.X), batch_size):
                X = X_shuffle[i: i + batch_size]
                y = y_shuffle[i: i + batch_size]
                dw, db = self.calc_derivative(X, y, self.w, self.b, regu)
                self.w -= self.lr * dw
                self.b -= self.lr * db
        return self.w, self.b

    def adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8, batch_size=1, regu = None):
        self._init_params()
        m_w = np.zeros_like(self.w)
        v_w = np.zeros_like(self.w)
        m_b = 0
        v_b = 0

        t = 0
        for epoch in range(self.n_iter):
            idxs = np.arange(self.n_samples)
            np.random.shuffle(idxs)
            X = self.X[idxs]
            y = self.y[idxs]
            for i in range(0, len(X), batch_size):
                mini_X = X[i : i + batch_size]
                mini_y = y[i : i + batch_size]

                dw, db = self.calc_derivative(mini_X, mini_y, self.w, self.b, regu)

                m_w = beta1 * m_w + (1 - beta1) * dw
                v_w = beta2 * v_w + (1 - beta2) * (dw * dw)

                m_b = beta1 * m_b + (1 - beta1) * db
                v_b = beta2 * v_b + (1 - beta2) * (db * db)
                t += 1
                m_hat_w = (m_w) / (1 - beta1 ** t)
                m_hat_b = (m_b) / (1 - beta1 ** t)
                v_hat_w = (v_w) / (1 - beta2 ** t)
                v_hat_b = (v_b) / (1 - beta2 ** t)

                self.w -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + epsilon)
                self.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + epsilon)
        return self.w, self.b