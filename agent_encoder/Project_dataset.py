import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
在加载数据集之前，使用
该方法是数据集准备 本项目使用该方法实现数据集加载
1. 读取深度图和碰撞图
2. 归一化深度图到[0, 255]范围
3. 添加噪声
4. 返回处理后的深度图和碰撞图，归一化到[0, 1]范围
"""

# 数据预处理函数
def preprocess_image(image):
    """将图像归一化到 [0.0, 1.0] 并转换为 PyTorch 张量"""
    image = image / 255.0
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


# 数据集类
class DepthCollisionDataset(Dataset):
    def __init__(self, depths_folder, colls_folder, transform=preprocess_image, return_file_name=False):
        self.depths_folder = depths_folder
        self.colls_folder = colls_folder
        self.transform = transform

        # 获取深度图和碰撞图的文件名列表
        self.depth_files = sorted([f for f in os.listdir(depths_folder) if f.endswith(".png")])
        self.coll_files = sorted([f for f in os.listdir(colls_folder) if f.endswith(".png")])

        self.IMAGE_SIZE = (480, 270)  # 图像大小
        self.MIN_DEPTH = int(0.2 * 255 / 10)  # 5
        self.MAX_DEPTH = 255
        self.return_file_name = return_file_name

        # 确保文件名一一对应
        assert len(self.depth_files) == len(self.coll_files), "深度图和碰撞图的数量不一致"
        for depth_file, coll_file in zip(self.depth_files, self.coll_files):
            assert depth_file == coll_file, f"文件名不匹配: {depth_file} vs {coll_file}"

    def __len__(self):
        return len(self.depth_files)

    # 添加深度值相关噪声的函数
    def add_depth_dependent_noise(self, depth_image, NOISE_FACTOR=0.05):
        """
        根据深度值添加噪声，深度值越大噪声越大。

        参数：
            depth_image (numpy.ndarray): 输入深度图，值范围 [0, 255]

        返回：
            noisy_depth_image (numpy.ndarray): 添加噪声后的深度图
        """
        # 计算标准差，深度值越大，标准差越大
        std_dev = (depth_image.astype(np.float32) / self.MAX_DEPTH) * (self.MAX_DEPTH - self.MIN_DEPTH) * NOISE_FACTOR

        # 生成高斯噪声
        noise = np.random.normal(0, std_dev, depth_image.shape)

        # 添加噪声并限制范围
        noisy_depth_image = np.clip(depth_image + noise, self.MIN_DEPTH, self.MAX_DEPTH)

        return noisy_depth_image.astype(np.uint8)

    def __getitem__(self, index):
        # 加载深度图和碰撞图
        depth_path = os.path.join(self.depths_folder, self.depth_files[index])
        coll_path = os.path.join(self.colls_folder, self.coll_files[index])

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        coll_image = cv2.imread(coll_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth_image = self.uint8_min_normalize(depth_image)

        # 0.5的概率添加噪声
        if np.random.rand() < 0.5:
            depth_image = self.add_depth_dependent_noise(depth_image)
        else:
            depth_image = depth_image

        # 归一化深度图
        depth_image = self.uint8_0_normalize(depth_image)

        # 应用转换
        if self.transform:
            # depth_image = self.transform(depth_image)
            depth_image = self.transform(depth_image)
            coll_image = self.transform(coll_image)

        if self.return_file_name:
            return depth_image, coll_image, self.depth_files[index]
        return depth_image, coll_image    # , self.depth_files[index]  # 返回文件名用于调试

    def uint8_0_normalize(self, depth_map):
        """
        将深度图归一化到[0, 255]范围，大于MAX_DEPTH设置为255 小于MIN_DEPTH的像素值设置为0
        返回值范围[0, 255]的深度图
        """
        depth_map[(depth_map <= self.MIN_DEPTH) & (depth_map > 0)] = 0
        return depth_map

    def uint8_min_normalize(self, normalized_depth_map):
        """
        将深度图归一化到[0, 255]范围，(大于0)&(小于MIN_DEPTH)的像素值设置为MIN_DEPTH
        返回值范围[0, 255]的深度图
        """
        max_val = np.max(normalized_depth_map)
        if max_val > 255:
            normalized_depth_map = normalized_depth_map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
        normalized_depth_map[(normalized_depth_map < self.MIN_DEPTH) & (normalized_depth_map > 0)] = self.MIN_DEPTH
        normalized_depth_map[normalized_depth_map > self.MAX_DEPTH] = self.MAX_DEPTH
        normalized_depth_map = cv2.resize(normalized_depth_map, self.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        return normalized_depth_map.astype(np.uint8)



if __name__ == "__main__":
    # 测试函数
    def test_dataset(dataloader):
        """测试数据集和数据加载器"""

        # 测试数据加载器
        print("\n测试数据加载器:")
        for batch in dataloader:
            depth_batch, coll_batch, filenames = batch

            # 显示批次中的前5个样本
            num_samples = min(5, len(filenames))
            plt.figure(figsize=(12, 3 * num_samples))

            for i in range(num_samples):
                plt.subplot(num_samples, 2, 2 * i + 1)
                plt.imshow(depth_batch[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.title(f"Noisy Depth - {filenames[i]}")
                plt.axis('off')

                plt.subplot(num_samples, 2, 2 * i + 2)
                plt.imshow(coll_batch[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                plt.title(f"Collision - {filenames[i]}")
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            # break  # 只显示一个批次

    # 数据集路径
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"

    # 创建数据集和数据加载器
    dataset = DepthCollisionDataset(
        depths_folder=depths_folder,
        colls_folder=colls_folder,
        transform=preprocess_image,
        return_file_name=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )

    # 测试数据集和数据加载器
    test_dataset(dataloader)