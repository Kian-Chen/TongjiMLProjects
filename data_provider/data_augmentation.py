import random
import numpy as np
import cv2

class DataAugmentation:
    def __init__(self, args):
        """
        初始化数据增强类
        :param args: 从命令行传入的参数
        """
        # 解析参数
        self.use_crop = args.use_crop
        self.use_scale = args.use_scale
        self.use_rotation = args.use_rotation
        self.use_flip = args.use_flip
        
        self.crop_size = tuple(map(int, args.crop_size.split(','))) if args.crop_size else None
        self.scale_range = tuple(map(float, args.scale_range.split(','))) if args.scale_range else (0.8, 1.2)
        self.rotation_range = tuple(map(int, args.rotation_range.split(','))) if args.rotation_range else (-30, 30)
        self.flip_prob = args.flip_prob

    def random_flip(self, img):
        """ 随机水平翻转 """
        if self.use_flip and random.random() < self.flip_prob:
            img = np.fliplr(img)
        return img

    def random_rotation(self, img):
            """ 随机旋转 """
            if self.use_rotation:
                angle = random.uniform(self.rotation_range[0], self.rotation_range[1])
                h, w = img.shape[:2]
                
                # 获取旋转矩阵
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                
                # 执行旋转，保持原始图像的大小
                if img.ndim == 3:  # 对于RGB或者多通道图像
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                else:  # 对于单通道图像
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                # 如果是单通道图像，确保其形状为 (H, W, 1)
                if img.ndim == 2:  # 灰度图像
                    img = img[:, :, np.newaxis]  # 添加通道维度
            return img

    def random_crop(self, img):
        """ 随机裁剪 """
        if self.use_crop and self.crop_size:
            h, w = img.shape[:2]
            crop_h, crop_w = self.crop_size
            top = random.randint(0, h - crop_h)
            left = random.randint(0, w - crop_w)
            img = img[top:top+crop_h, left:left+crop_w]
            # 裁剪后强制调整图像大小到原始输入大小
            img = cv2.resize(img, (w, h))
        return img

    def random_scaling(self, img):
        """ 随机缩放 """
        if self.use_scale:
            scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
            h, w = img.shape[:2]
            
            # 计算缩放后的尺寸
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # 使用 OpenCV 进行缩放，保留通道信息
            img = cv2.resize(img, (new_w, new_h))
            
            # 缩放后强制调整图像大小到原始输入大小，使用相同的插值方法
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 如果是单通道图像，确保其形状为 (H, W, 1)
            if img.ndim == 2:  # 灰度图像
                img = img[:, :, np.newaxis]  # 添加通道维度
        return img

    def augment(self, X):
        """
        对数据集 X 进行增强操作
        :param X: 输入数据，假设是 (N, H, W, C) 的 numpy 数组，其中 N 是样本数，H, W 是图像大小，C 是通道数
        :return: 增强后的数据
        """
        augmented_images = []
        for img in X:
            img = self.random_flip(img)
            img = self.random_rotation(img)
            img = self.random_crop(img)
            img = self.random_scaling(img)
            augmented_images.append(img)
        
        return np.array(augmented_images)

