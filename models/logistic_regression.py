import numpy as np
from models.base_model import BaseModel

class LogisticRegression(BaseModel):
    def __init__(self, args):
        super().__init__(args=args)
        self.penalty = args.lr_penalty
        self.learning_rate = args.learning_rate
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.weights = None
        self.bias = None
        self.num_classes = 100 if(args.data == 'cifar-100') else 10

    def _softmax(self, z):
        """Compute the softmax function."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, y_true, y_pred):
        """Compute the cross-entropy loss."""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.log(y_pred[np.arange(len(y_true)), y_true]))
    
    def _gradient_descent(self, X, y_true, y_pred):
        """Compute the gradients and update the weights and bias."""
        m = X.shape[0]
        grad_w = (1 / m) * np.dot(X.T, (y_pred - np.eye(self.num_classes)[y_true]))
        grad_b = (1 / m) * np.sum(y_pred - np.eye(self.num_classes)[y_true], axis=0)
        if self.penalty == 'l2':
            grad_w += (self.learning_rate / m) * self.weights
        elif self.penalty == 'l1':
            grad_w += (self.learning_rate / m) * np.sign(self.weights)
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b
        
    def train(self, X_train, y_train):
        X_train = self._preprocess_images(X_train, flag='train')
        num_features = X_train.shape[1]
        self.weights = np.zeros((num_features, self.num_classes))
        self.bias = np.zeros(self.num_classes)

        for epoch in range(self.num_epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]

                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._softmax(z)

                loss = self._cross_entropy_loss(y_batch, y_pred)
                self._gradient_descent(X_batch, y_batch, y_pred)
    def predict(self, X):
        X = self._preprocess_images(X, flag='test')
        z = np.dot(X, self.weights) + self.bias
        y_pred = self._softmax(z)
        return np.argmax(y_pred, axis=1)

