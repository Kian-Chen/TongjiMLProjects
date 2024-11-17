import pickle
import os
import abc
from sklearn.decomposition import PCA


class BaseModel(abc.ABC):
    """
    所有模型的基类，提供通用的训练、预测接口
    """
    def __init__(self, args):
        self.use_pca = args.use_pca
        self.n_components = args.pca_components
        if args.use_pca:
            self.pca = PCA(n_components=self.n_components)


    @abc.abstractmethod
    def train(self, X_train, y_train):
        """
        训练模型
        :param X_train: 训练数据特征
        :param y_train: 训练数据标签
        """
        pass

    @abc.abstractmethod
    def predict(self, X):
        """
        进行预测
        :param X: 输入数据
        :return: 预测结果
        """
        pass

    def _preprocess_images(self, X, flag='train'):
        """
        将图像数据 (N, H, W, C) 转换为 (N, H*W*C)
        """
        N, H, W, C = X.shape
        X_flat = X.reshape(N, -1)
        if self.use_pca:
            if flag == 'train':
                X_flat = self.pca.fit_transform(X_flat)
            else:
                X_flat = self.pca.transform(X_flat)
        return X_flat

    def save(self, file_path):
        """
        保存模型到指定路径
        :param file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"模型已保存到 {file_path}")

    def load(file_path):
        """
        从指定路径加载模型
        :param file_path: 加载路径
        :return: 加载的模型对象
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已从 {file_path} 加载")
        return model
