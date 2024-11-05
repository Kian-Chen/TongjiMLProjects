from models.base_model import BaseModel
import numpy as np

class KNNClassifier(BaseModel):
    def __init__(self, k_neighbors=3):
        super().__init__()
        self.k_neighbors = k_neighbors

    def train(self, X_train, y_train, learning_rate=None):
        # 训练逻辑，核心部分先pass
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 预测逻辑，核心部分先pass
        pass

    def evaluate(self, X, y):
        # 评估逻辑，核心部分先pass
        return 0.0

    def save(self, filepath):
        # 保存模型逻辑，核心部分先pass
        pass
