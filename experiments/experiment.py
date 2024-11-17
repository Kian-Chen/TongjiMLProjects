from models import bayesian, svc, knn, logistic_regression
from data_provider.data_factory import data_provider
from data_provider.data_augmentation import DataAugmentation
from utils.metrics import evaluate_classification
from utils.visualization import plot_confusion_matrix
import time
import os

class Experiment:
    def __init__(self, args):
        # 解析 args 并初始化相应的属性
        self.args = args
        self.model_type = args.model
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.checkpoints = args.checkpoints
        log_dir = args.log_dir
        self.log_file = os.path.join(log_dir, args.log_file)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.checkpoints):
            os.makedirs(self.checkpoints)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.use_augmentation = args.augmentation
        self.random_state = args.random_state

        self.model = self._initialize_model()
        self.data_loader = data_provider(self.args)

    def _initialize_model(self):
        model_dict = {
            'bayesian': bayesian.BayesianClassifier,
            'svc': svc.SVCClassifier,
            'knn': knn.KNNClassifier,
            'logistic_regression': logistic_regression.LogisticRegression
        }
        try:
            model = model_dict[self.model_type](self.args)
        except KeyError:
            raise ValueError(f"Invalid model type: {self.model_type}")
        return model

    def _prepare_data(self):
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_loader.prepare_datasets()

        if self.use_augmentation:
            augmenter = DataAugmentation(self.args)
            X_train = augmenter.augment(X_train)
            X_valid = augmenter.augment(X_valid)
            X_test = augmenter.augment(X_test)

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def run(self, setting):
        # 执行实验的主流程
        print(f"Starting experiment with {self.model_type} model")
        
        # 准备数据
        X_train, X_valid, X_test, y_train, y_valid, y_test = self._prepare_data()

        # 训练模型
        print("Training the model...")
        start_time = time.time()
        self.model.train(X_train, y_train)
        print(f"Training completed in {time.time() - start_time:.2f} seconds.")

        # 评估模型
        print("Evaluating the model...")
        train_pred = self.model.predict(X_train)
        valid_pred = self.model.predict(X_valid)
        test_pred = self.model.predict(X_test)

        train_result = evaluate_classification(y_train, train_pred)
        valid_result = evaluate_classification(y_valid, valid_pred)
        test_result = evaluate_classification(y_test, test_pred)

        print(f"Training accuracy: {train_result['accuracy']:.4f}")
        print(f"Validation accuracy: {valid_result['accuracy']:.4f}")
        print(f"Test accuracy: {test_result['accuracy']:.4f}")

        if not os.path.exists(os.path.join(self.checkpoints, setting)):
            os.makedirs(os.path.join(self.checkpoints, setting))
        self.model.save(os.path.join(self.checkpoints, setting, "checkpoints.pth"))


        with open(self.log_file, 'a') as f:
            f.write(setting + "  \n")
            res_str = f"Acc: {test_result['accuracy']:.4f}, Prec: {test_result['precision']:.4f}, Rec: {test_result['recall']:.4f}, F1: {test_result['f1']:.4f}"
            f.write(res_str)
            f.write('\n\n')

        plot_confusion_matrix(cm=test_result['confusion_matrix'],
                              save_path=os.path.join(self.save_dir, setting))
