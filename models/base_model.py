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
    
    @abc.abstractmethod
    def evaluate(self, X, y):
        """
        评估模型表现
        :param X: 测试数据特征
        :param y: 测试数据标签
        :return: 模型准确率
        """
        pass
    
    @abc.abstractmethod
    def save(self, filepath):
        """
        保存模型
        :param filepath: 模型保存路径
        """
        pass
