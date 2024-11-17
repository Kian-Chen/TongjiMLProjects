import pickle
import os
import abc


class BaseModel(abc.ABC):
    """
    所有模型的基类，提供通用的训练、预测接口
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def train(self, X_train, y_train, learning_rate):
        """
        训练模型
        :param X_train: 训练数据特征
        :param y_train: 训练数据标签
        :param learning_rate: 学习率
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

    def save(self, file_path):
        """
        保存模型到指定路径
        :param file_path: 保存路径
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"模型已保存到 {file_path}")

    @staticmethod
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
