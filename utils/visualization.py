import matplotlib.pyplot as plt
import os
import seaborn as sns

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
