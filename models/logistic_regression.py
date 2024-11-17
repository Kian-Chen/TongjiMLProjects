from models.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, args):
        super().__init__(args=args)
        self.penalty = args.penalty

    def train(self, X_train, y_train):
        X_train = self._preprocess_images(X_train, flag='train')
        # 训练逻辑，核心部分先pass
        pass

    def predict(self, X):
        X = self._preprocess_images(X, flag='test')
        # 预测逻辑，核心部分先pass
        pass

