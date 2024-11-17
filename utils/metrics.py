from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import numpy as np

def calculate_accuracy(y_true, y_pred):
    """
    计算准确率
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 准确率
    """
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred):
    """
    计算精确度（Precision）
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 精确度
    """
    return precision_score(y_true, y_pred, average='weighted')

def calculate_recall(y_true, y_pred):
    """
    计算召回率（Recall）
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 召回率
    """
    return recall_score(y_true, y_pred, average='weighted')

def calculate_f1(y_true, y_pred):
    """
    计算F1分数
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: F1分数
    """
    return f1_score(y_true, y_pred, average='weighted')

def calculate_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 混淆矩阵
    """
    return confusion_matrix(y_true, y_pred)

def calculate_roc_auc(y_true, y_pred_proba):
    """
    计算ROC AUC分数
    :param y_true: 真实标签
    :param y_pred_proba: 预测的类别概率（通常是通过模型的`predict_proba`获得）
    :return: AUC分数
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])  # 计算二分类的fpr、tpr
    return auc(fpr, tpr)

def evaluate_classification(y_true, y_pred, y_pred_proba=None):
    """
    计算并返回多个评估指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param y_pred_proba: 预测的概率（仅用于计算AUC，二分类才使用）
    :return: 各种评估指标的字典
    """
    metrics = {
        'accuracy': calculate_accuracy(y_true, y_pred),
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1': calculate_f1(y_true, y_pred),
        'confusion_matrix': calculate_confusion_matrix(y_true, y_pred)
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = calculate_roc_auc(y_true, y_pred_proba)

    return metrics
