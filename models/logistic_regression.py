from models.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty

    def train(self, X_train, y_train, learning_rate=None):
        # 训练逻辑，核心部分先pass
        pass

    def predict(self, X):
        # 预测逻辑，核心部分先pass
        pass

