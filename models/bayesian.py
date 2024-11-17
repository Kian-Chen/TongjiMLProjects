from models.base_model import BaseModel
import numpy as np
from sklearn.decomposition import PCA

class BayesianClassifier(BaseModel):
    def __init__(self, args):
        """
        初始化类，增加use_sklearn参数来选择是否使用sklearn的朴素贝叶斯作为标准
        同时加入PCA降维的参数n_components
        """
        super().__init__(args=args)
        self.var_smoothing = args.bayesian_var_smoothing
        self.class_probabilities = None
        self.means = None
        self.vars = None


    def train(self, X_train, y_train):
        """
        训练朴素贝叶斯模型：计算每个类别的先验概率、均值和方差
        如果use_sklearn为True，则使用sklearn的朴素贝叶斯模型
        在训练前进行PCA降维到指定的维度
        """
        X_train = self._preprocess_images(X_train, flag='train')
        num_samples = X_train.shape[0]

        num_classes = len(np.unique(y_train))

        # 初始化类的均值、方差和先验概率
        self.class_probabilities = np.zeros(num_classes)
        self.means = np.zeros((num_classes, self.n_components))  # 使用降维后的维度
        self.vars = np.zeros((num_classes, self.n_components))  # 使用降维后的维度

        # 计算每个类别的先验概率 P(C)
        for c in range(num_classes):
            class_samples = X_train[y_train == c]
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
        X = self._preprocess_images(X, flag='test')
        num_samples = X.shape[0]

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
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var) + ((X - mean) ** 2) / var, axis=1)

            # 计算后验概率：log(P(C|x)) = log(P(C)) + log(P(x|C))
            posterior_probs[:, c] = prior + log_likelihood

        # 返回每个样本的最大后验概率的类别
        return np.argmax(posterior_probs, axis=1)


