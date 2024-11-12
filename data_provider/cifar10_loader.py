import os
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image

class CIFAR10Loader:
    def __init__(self, data_dir='data'):
        """
        初始化CIFAR-10数据加载器，使用torchvision自动下载并加载数据集。
        :param data_dir: 数据集保存路径
        """
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((32, 32))  # 确保图像大小一致
        ])

    def load_data(self):
        """
        使用 torchvision 加载 CIFAR-10 数据集
        :return: X, y 数据特征和标签
        """
        # 加载训练集和测试集
        train_dataset = CIFAR10(root=self.data_dir, train=True, download=True, transform=self.transform)
        test_dataset = CIFAR10(root=self.data_dir, train=False, download=True, transform=self.transform)

        # 获取数据和标签，转换为numpy格式
        X_train = np.array([np.array(train_dataset[i][0]) for i in range(len(train_dataset))])
        y_train = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

        X_test = np.array([np.array(test_dataset[i][0]) for i in range(len(test_dataset))])
        y_test = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

        # 数据归一化（将像素值归一化到0-1之间）
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        return X_train, y_train, X_test, y_test

    def prepare_datasets(self, test_size=0.2, valid_size=0.2, random_state=42):
        """
        加载数据并进行分割，生成训练集、验证集和测试集。
        :param test_size: 测试集占比
        :param valid_size: 验证集占比
        :param random_state: 随机种子
        :return: 分割后的训练集、验证集、测试集的特征和标签
        """
        # 加载数据
        X_train, y_train, X_test, y_test = self.load_data()

        # 划分训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state)

        return X_train, X_valid, X_test, y_train, y_valid, y_test