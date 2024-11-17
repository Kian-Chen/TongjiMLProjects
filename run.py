from utils.get_args import parse_args
from experiments.experiment import Experiment

<<<<<<< HEAD
=======
def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Classification Experiment using Traditional Machine Learning Models')

    # Basic Parameters
    parser.add_argument('--model', type=str, choices=['bayesian', 'svc', 'knn', 'logistic_regression'],
                        default='logistic_regression', help='选择模型：bayesian, svc, knn, logistic_regression')
    parser.add_argument('--data', type=str, choices=['mnist', 'cifar-10', 'cifar-100'],
                        default='cifar-10', help='Choose dataset: mnist, cifar-100...')

    # Data Augmentation
    parser.add_argument('--augmentation', type=bool, default=True, help='whether to use data augmentation or not')
    parser.add_argument('--use_crop', type=bool, default=False, help='Whether to use random crop')
    parser.add_argument('--use_scale', type=bool, default=True, help='Whether to use random scaling')
    parser.add_argument('--use_rotation', type=bool, default=False, help='Whether to use random rotation')
    parser.add_argument('--use_flip', type=bool, default=False, help='Whether to use random flip')

    parser.add_argument('--crop_size', type=str, default=None, help='Crop size (height, width), e.g., "28,28"')
    parser.add_argument('--scale_range', type=str, default="0.8,1.2", help='Scaling range, default is (0.8, 1.2)')
    parser.add_argument('--rotation_range', type=str, default="-30,30", help='Rotation angle range, default is (-30, 30)')
    parser.add_argument('--flip_prob', type=float, default=0.5, help='Probability of horizontal flip, default is 0.5')


    # Path Parameters
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='the directory of checkpoints')
    parser.add_argument('--data_dir', type=str, default='./datasets/', help='the directory of dataset')
    parser.add_argument('--save_dir', type=str, default='./results/', help='where to save the results')
    parser.add_argument('--log_dir', type=str, default='./logs/', help='where to save the logs')
    parser.add_argument('--log_file', type=str, default='log.txt', help='log file')

    # PCA Parameters
    parser.add_argument('--use_pca', type=bool, default=True, help='whether to use PCA or not')
    parser.add_argument('--pca_components', type=int, default=21, help='number of PCA components to keep')


    # Training Parameters
    parser.add_argument('--itr', type=int, default=5, help='iterations')

    # 对于KNN模型的超参数
    parser.add_argument('--k_neighbors', type=int, default=3, help='KNN中邻居的数量')

    # 对于SVC模型的超参数
    parser.add_argument('--svc_kernel', type=str, choices=['linear', 'poly', 'rbf'], default='linear', help='SVM核函数类型')
    parser.add_argument('--svc_C', type=float, default=1.0, help='SVM惩罚参数C')
    parser.add_argument('--svc_max_iter', type=int, default=1000, help='SVM最大迭代次数')
    parser.add_argument('--svc_tol', type=float, default=1e-3, help='SVM容忍度')


    # 对于逻辑斯蒂回归的超参数
    parser.add_argument('--lr_penalty', type=str, choices=['l2', 'l1'], default='l2', help='逻辑斯蒂回归的惩罚项')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    
    # 对于贝叶斯分类器的超参数
    parser.add_argument('--bayesian_var_smoothing', type=float, default=1e-9, help='贝叶斯分类器的方差平滑参数')

    parser.add_argument('--random_state', type=int, default=42, help='随机种子')

    return parser.parse_args()
>>>>>>> e93ac56d8a7e28c9982ca26a4074dc6499dc938d

def main():
    args = parse_args()

    experiment = Experiment(args)

    for ii in range(args.itr):
        setting = '{}_{}_pca{}{}_crop{}_scale{}_rotate{}_filp{}_Exp_{}'.format(
            args.model,
            args.data,
            args.use_pca,
            args.pca_components,
            args.use_crop,
            args.use_scale,
            args.use_rotation,
            args.use_flip,
            ii
        )
        experiment.run(setting)

if __name__ == "__main__":
    main()
