from sklearn.metrics import accuracy_score

def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确率
    """
    return accuracy_score(y_true, y_pred)
