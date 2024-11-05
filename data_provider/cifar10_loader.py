import os
import tarfile
import pickle
import numpy as np
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

class CIFAR10Loader:
    def __init__(self, dataset_path='data/raw/cifar-10-python.tar.gz', download_url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
        """
        初始化CIFAR-10数据加载器，自动检查、下载并解压数据集
        :param dataset_path: 数据集下载后的保存路径
        :param download_url: CIFAR-10数据集的下载地址
        """
        self.dataset_path = dataset_path
        self.download_url = download_url
        self.data_dir = os.path.dirname(self.dataset_path)

        # 自动下载并解压数据集（如果数据集尚未存在）
        self.download_and_extract()

    def download_and_extract(self):
        """
        下载并解压CIFAR-10数据集
        """
        # 确保数据存放目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)  # 如果不存在，创建目录
        
        # 如果文件已经存在，直接跳过下载
        if not os.path.exists(self.dataset_path):
            print(f"Downloading CIFAR-10 dataset from {self.download_url}...")
            urlretrieve(self.download_url, self.dataset_path)
            print(f"Dataset downloaded and saved to {self.dataset_path}")
        
        # 解压数据
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(self.dataset_path, 'r:gz') as tar:
            tar.extractall(path=self.data_dir)
        print("Extraction completed.")

    def load_data(self):
        """
        加载CIFAR-10数据集
        :return: X, y 数据特征和标签
        """
        # CIFAR-10 数据集解压后包含多个二进制文件，需要手动加载
        data_files = [
            'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch'
        ]
        data = []
        labels = []

        for file_name in data_files:
            file_path = os.path.join(self.data_dir, 'cifar-10-batches-py', file_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')  # 使用bytes解码文件内容
                data.append(batch[b'data'])
                labels.append(batch[b'labels'])

        # 拼接数据和标签
        X = np.vstack(data)
        y = np.concatenate(labels)

        # 数据归一化（将像素值归一化到0-1之间）
        X = X.astype(np.float32) / 255.0
        
        return X, y

    def prepare_datasets(self, test_size=0.2, valid_size=0.2, random_state=42):
        """
        加载数据并进行分割，生成训练集、验证集和测试集
        :param test_size: 测试集占比
        :param valid_size: 验证集占比
        :param random_state: 随机种子
        :return: 分割后的训练集、验证集、测试集的特征和标签
        """
        # 加载数据
        X, y = self.load_data()

        # 划分训练集与测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # 在训练集上划分验证集
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=random_state)

        return X_train, X_valid, X_test, y_train, y_valid, y_test
