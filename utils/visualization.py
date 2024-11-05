import matplotlib.pyplot as plt

def plot_metrics(train_accuracy, valid_accuracy, test_accuracy):
    """
    绘制模型准确率曲线
    :param train_accuracy: 训练集准确率
    :param valid_accuracy: 验证集准确率
    :param test_accuracy: 测试集准确率
    """
    epochs = range(1, 2)  # 这里只是示例，可以扩展

    plt.plot(epochs, [train_accuracy], label='Train Accuracy')
    plt.plot(epochs, [valid_accuracy], label='Validation Accuracy')
    plt.plot(epochs, [test_accuracy], label='Test Accuracy')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
