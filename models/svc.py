import numpy as np
from models.base_model import BaseModel

class SVCClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args=args)
        self.kernel = args.svc_kernel
        self.C = args.svc_C
        self.max_iter = args.svc_max_iter
        self.tol = args.svc_tol
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.Ws = None
        self.bs = None

    def linear_kernel(self, X1, X2):
        """ 
        线性核函数
        """
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2, gamma=0.1):
        """ 
        高斯RBF核函数
        """
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        return np.exp(-gamma * sq_dists)

    def kernel_function(self, X1, X2):
        """ 
        选择核函数
        """
        if self.kernel == 'linear':
            return self.linear_kernel(X1, X2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError("未知的核函数")

    def train(self, X_train, y_train):
        """ 
        使用One-vs-Rest方法训练多分类SVM
        """
        X_train = self._preprocess_images(X_train, flag='train')
        m, n = X_train.shape
        y_train = np.array(y_train)

        unique_classes = np.unique(y_train)
        num_classs = len(unique_classes)

        self.alphas = []
        self.Ws = []
        self.bs = []

        for c in range(num_classs):
            y_binary = np.where(y_train == c, 1, -1)

            alpha = np.zeros(m)
            W = np.zeros(n)
            b = 0

            for iter in range(self.max_iter):
                alpha_prev = np.copy(alpha)
                for i in range(m):
                    Ei = self._compute_error(X_train, y_binary, i, W, b)
                    
                    if (y_binary[i] * Ei < -self.tol and alpha[i] < self.C) or (y_binary[i] * Ei > self.tol and alpha[i] > 0):
                        j = self._select_j(i, m)
                        Ej = self._compute_error(X_train, y_binary, j, W, b)

                        alpha_i_old, alpha_j_old = alpha[i], alpha[j]

                        L, H = self._compute_L_H(alpha, i, j, y_binary)

                        eta = 2.0 * self.kernel_function(X_train[i], X_train[j]) - self.kernel_function(X_train[i], X_train[i]) - self.kernel_function(X_train[j], X_train[j])
                        if eta >= 0:
                            continue

                        alpha[j] -= y_binary[j] * (Ei - Ej) / eta
                        alpha[j] = np.clip(alpha[j], L, H)

                        if abs(alpha[j] - alpha_j_old) < self.tol:
                            continue

                        alpha[i] += y_binary[i] * y_binary[j] * (alpha_j_old - alpha[j])

                        b1 = b - Ei - y_binary[i] * (alpha[i] - alpha_i_old) * self.kernel_function(X_train[i], X_train[i]) - y_binary[j] * (alpha[j] - alpha_j_old) * self.kernel_function(X_train[i], X_train[j])
                        b2 = b - Ej - y_binary[i] * (alpha[i] - alpha_i_old) * self.kernel_function(X_train[i], X_train[j]) - y_binary[j] * (alpha[j] - alpha_j_old) * self.kernel_function(X_train[j], X_train[j])

                        if 0 < alpha[i] < self.C:
                            b = b1
                        elif 0 < alpha[j] < self.C:
                            b = b2
                        else:
                            b = (b1 + b2) / 2

                diff = np.linalg.norm(alpha - alpha_prev)
                if diff < self.tol:
                    print(f"类别 {c} 在 {iter+1} 次迭代后收敛")
                    break

            W = np.dot((alpha * y_binary).T, X_train)

            self.alphas.append(alpha)
            self.Ws.append(W)
            self.bs.append(b)

            support_vector_indices = alpha > 1e-4
            self.support_vectors = X_train[support_vector_indices]
            self.support_vector_labels = y_train[support_vector_indices]

    def _compute_error(self, X_train, y_train, i, W, b):
        """ 
        计算第i个样本的误差 
        """
        return np.dot(X_train[i], W) + b - y_train[i]

    def _select_j(self, i, m):
        """ 
        随机选择另一个j 
        """
        j = i
        while j == i:
            j = np.random.randint(0, m)
        return j

    def _compute_L_H(self, alpha, i, j, y_train):
        """ 
        计算L和H，用于alpha的更新 
        """
        if y_train[i] != y_train[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(self.C, self.C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - self.C)
            H = min(self.C, alpha[i] + alpha[j])
        return L, H

    def predict(self, X):
        """ 
        预测 
        """
        X = self._preprocess_images(X, flag='test')
        if self.Ws is None:
            raise ValueError("模型未被训练")

        predictions = np.array([np.dot(X, W) + b for W, b in zip(self.Ws, self.bs)])
        return np.argmax(predictions, axis=0)
