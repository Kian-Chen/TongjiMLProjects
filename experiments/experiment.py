from models import bayesian, svc, knn, logistic_regression
from data_provider.cifar10_loader import CIFAR10Loader
from data_provider.data_augmentation import DataAugmentation
from utils import metrics, visualization
import time
import os

class Experiment:
    def __init__(self, args):
        # 解析 args 并初始化相应的属性
        self.model_type = args.model
        self.learning_rate = args.learning_rate
        self.k_neighbors = args.k_neighbors
        self.svc_kernel = args.svc_kernel
        self.lr_penalty = args.lr_penalty
        self.bayesian_var_smoothing = args.bayesian_var_smoothing
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.use_augmentation = args.augmentation
        self.random_state = args.random_state

        # 初始化模型
        self.model = self._initialize_model()

    def _initialize_model(self):
        # 根据模型类型初始化相应的模型
        if self.model_type == 'bayesian':
            return bayesian.BayesianClassifier(var_smoothing=self.bayesian_var_smoothing)
        elif self.model_type == 'svc':
            return svc.SVCClassifier(kernel=self.svc_kernel)
        elif self.model_type == 'knn':
            return knn.KNNClassifier(k_neighbors=self.k_neighbors)
        elif self.model_type == 'logistic_regression':
            return logistic_regression.LogisticRegression(penalty=self.lr_penalty)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _prepare_data(self):
        # 加载并处理数据
        data_loader = CIFAR10Loader()
        X, y = data_loader.load_data()
        X_train, X_valid, X_test, y_train, y_valid, y_test = data_loader.prepare_datasets()

        if self.use_augmentation:
            augmenter = DataAugmentation()
            X_train = augmenter.augment(X_train)
            X_valid = augmenter.augment(X_valid)
            X_test = augmenter.augment(X_test)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def run(self):
        # 执行实验的主流程
        print(f"Starting experiment with {self.model_type} model")
        
        # 准备数据
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._prepare_data()

        # 训练模型
        print("Training the model...")
        start_time = time.time()
        self.model.train(X_train, y_train, learning_rate=self.learning_rate)
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

        # 评估模型
        print("Evaluating the model...")
        train_accuracy = self.model.evaluate(X_train, y_train)
        valid_accuracy = self.model.evaluate(X_valid, y_valid)
        test_accuracy = self.model.evaluate(X_test, y_test)

        # 打印评估结果
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Validation accuracy: {valid_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")

        # 保存结果
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 保存模型（如果需要）
        self.model.save(os.path.join(self.save_dir, f"{self.model_type}_model.pth"))

        # 可视化（如有需要）
        visualization.plot_metrics(train_accuracy, valid_accuracy, test_accuracy)
