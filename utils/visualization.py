import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


def plot_confusion_matrix(cm, labels=None, title="Confusion Matrix", 
                          save_path='./results/', filename='confusion_matrix.pdf'):
    """
    绘制混淆矩阵的热力图
    :param cm: 混淆矩阵 (numpy array or list)
    :param labels: 类别标签列表 (默认为 None)
    :param title: 图的标题 (默认为 "Confusion Matrix")
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if labels is None:
        labels = [str(i) for i in range(len(cm))]
    
    plt.figure(figsize=(8, 6))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()

def plot_tsne(X_pca, y, tsne_perplexity=30, 
              tsne_iter=1000, save_dir='./results/',
              filename='tsne.pdf'):
    """
    使用 t-SNE 可视化已降维的数据。
    
    :param X_pca: PCA 降维后的数据
    :param y: 标签，形状为 (N,)
    :param tsne_perplexity: t-SNE 的 perplexity 参数
    :param tsne_iter: t-SNE 的迭代次数
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # t-SNE 降维
    print("正在进行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=tsne_iter, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    print("t-SNE 降维完成。")

    # 绘制 t-SNE 图
    plt.figure(figsize=(10, 8))
    num_classes = len(np.unique(y))
    colors = plt.cm.get_cmap("tab10", num_classes)

    for class_idx in range(num_classes):
        indices = y == class_idx
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=f"Class {class_idx}", alpha=0.7, s=15)

    plt.title("t-SNE Visualization")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.show()


if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.get_args import parse_args
    from data_provider.data_factory import data_provider

    args = parse_args()
    data_set = data_provider(args)
    X_train, y_train, X_test, y_test = data_set.load_data()

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    X_flat = X.reshape(X.shape[0], -1)

    # PCA
    pca = PCA(n_components=args.pca_components)
    X_pca = pca.fit_transform(X_flat)

    plot_tsne(X_pca, y, 
              tsne_perplexity=args.tsne_perplexity, 
              tsne_iter=args.tsne_iter, 
              save_dir=os.path.join(args.save_dir, args.data))