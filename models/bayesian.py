from models.base_model import BaseModel
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

class BayesianClassifier(BaseModel):
    def __init__(self, var_smoothing=1e-9, n_components=21, use_sklearn=False):
        """
        初始化类，增加use_sklearn参数来选择是否使用sklearn的朴素贝叶斯作为标准
        同时加入PCA降维的参数n_components
        """
        super().__init__()
        self.var_smoothing = var_smoothing
        self.class_probabilities = None
        self.means = None
        self.vars = None
        self.use_sklearn = use_sklearn
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

        if self.use_sklearn:
            self.sklearn_model = GaussianNB()

    def train(self, X_train, y_train, learning_rate=None):
        """
        训练朴素贝叶斯模型：计算每个类别的先验概率、均值和方差
        如果use_sklearn为True，则使用sklearn的朴素贝叶斯模型
        在训练前进行PCA降维到指定的维度
        """
        num_samples, height, width, channels = X_train.shape
        num_features = height * width * channels  # 32 * 32 * 3 = 3072 个特征
        X_train_flat = X_train.reshape(num_samples, num_features)

        # 使用PCA进行降维
        X_train_flat = self.pca.fit_transform(X_train_flat)  # 降维到n_components维

        if self.use_sklearn:
            self.sklearn_model.fit(X_train_flat, y_train)
        else:
            num_classes = len(np.unique(y_train))

            # 初始化类的均值、方差和先验概率
            self.class_probabilities = np.zeros(num_classes)
            self.means = np.zeros((num_classes, self.n_components))  # 使用降维后的维度
            self.vars = np.zeros((num_classes, self.n_components))  # 使用降维后的维度

            # 计算每个类别的先验概率 P(C)
            for c in range(num_classes):
                class_samples = X_train_flat[y_train == c]
                self.class_probabilities[c] = len(class_samples) / num_samples

                # 计算每个类别下的特征均值和方差 P(x_i | C) 使用高斯分布
                self.means[c] = np.mean(class_samples, axis=0)
                self.vars[c] = np.var(class_samples, axis=0) + self.var_smoothing  # 加上平滑项

    def predict(self, X):
        """
        使用训练好的朴素贝叶斯模型对测试数据进行预测
        如果use_sklearn为True，则使用sklearn的朴素贝叶斯模型
        在预测前进行PCA降维到指定的维度
        """
        num_samples = X.shape[0]
        X_flat = X.reshape(num_samples, -1)  # 变成 (num_samples, 3072)

        # 使用PCA进行降维
        X_flat = self.pca.transform(X_flat)  # 降维到n_components维

        if self.use_sklearn:
            # 使用sklearn模型进行预测
            return self.sklearn_model.predict(X_flat)
        else:
            # 自己实现的朴素贝叶斯预测
            num_classes = self.class_probabilities.shape[0]

            # 初始化一个数组来存储每个测试样本在每个类别下的后验概率
            posterior_probs = np.zeros((num_samples, num_classes))

            # 对于每个测试样本，计算其在每个类别下的后验概率
            for c in range(num_classes):
                # 计算 P(x | C) 使用高斯分布公式
                mean = self.means[c]
                var = self.vars[c]
                prior = np.log(self.class_probabilities[c])

                # 对每个特征计算概率，使用高斯分布
                log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X_flat - mean) ** 2) / var, axis=1)

                # 计算后验概率：log(P(C|x)) = log(P(C)) + log(P(x|C))
                posterior_probs[:, c] = prior + log_likelihood

            # 返回每个样本的最大后验概率的类别
            return np.argmax(posterior_probs, axis=1)


