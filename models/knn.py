from models.base_model import BaseModel
import numpy as np
import torch

class KNNClassifier(BaseModel):
    def __init__(self, args):
        super().__init__(args=args)
        self.k_neighbors = args.k_neighbors

    def train(self, X_train, y_train):
        # 训练逻辑，核心部分先pass
        X_train = self._preprocess_images(X_train, flag='train')
        self.X_train = torch.tensor(X_train)
        self.y_train = torch.tensor(y_train)

    def predict(self, X):
        X = self._preprocess_images(X, flag='test')
        distances = KNNClassifier.compute_distances_no_loops(self.X_train, torch.tensor(X))
        y_test_pred = KNNClassifier.predict_labels(distances, self.y_train, self.k_neighbors)
        return y_test_pred

    @staticmethod
    def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):

        num_train = x_train.shape[0]
        num_test = x_test.shape[0]
        dists = x_train.new_zeros(num_train, num_test)

        dists.to(x_train.dtype)

        train_sum_flatten = x_train.square().sum(dim=1).view(num_train, -1)
        test_sum_flatten = x_test.square().sum(dim=1).view(-1, num_test)

        dists = train_sum_flatten + test_sum_flatten - 2 * torch.mm(x_train, x_test.t())

        return dists
    @staticmethod
    def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):

        num_train, num_test = dists.shape
        y_pred = torch.zeros(num_test, dtype=torch.int64)

        values, index = torch.topk(dists, largest=False, dim=0, k=k)
        y_pred = y_train[index].mode(dim=0).values

        return y_pred