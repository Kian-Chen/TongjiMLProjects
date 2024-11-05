import argparse
from experiments.experiment import Experiment

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification Experiment using Traditional Machine Learning Models')

    # 模型选择
    parser.add_argument('--model', type=str, choices=['bayesian', 'svc', 'knn', 'logistic_regression'],
                        default='bayesian', help='选择模型：bayesian, svc, knn, logistic_regression')

    # 数据处理相关超参数
    parser.add_argument('--data_dir', type=str, default='data/raw', help='数据集路径')
    parser.add_argument('--save_dir', type=str, default='experiments/results/', help='实验结果保存路径')

    # 训练相关超参数
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')

    # 对于KNN模型的超参数
    parser.add_argument('--k_neighbors', type=int, default=3, help='KNN中邻居的数量')

    # 对于SVC模型的超参数
    parser.add_argument('--svc_kernel', type=str, choices=['linear', 'poly', 'rbf'], default='linear', help='SVM核函数类型')

    # 对于逻辑斯蒂回归的超参数
    parser.add_argument('--lr_penalty', type=str, choices=['l2', 'l1'], default='l2', help='逻辑斯蒂回归的惩罚项')

    # 对于贝叶斯分类器的超参数
    parser.add_argument('--bayesian_var_smoothing', type=float, default=1e-9, help='贝叶斯分类器的方差平滑参数')

    # 是否使用数据增强
    parser.add_argument('--augmentation', action='store_true', help='是否使用数据增强')

    parser.add_argument('--random_state', type=int, default=42, help='随机种子')

    return parser.parse_args()

def main():
    args = parse_args()

    experiment = Experiment(args)

    experiment.run()

if __name__ == "__main__":
    main()
