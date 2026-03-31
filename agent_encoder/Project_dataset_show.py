import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

"""
在加载数据集之前，使用
该方法是数据集准备 本项目使用该方法实现数据集加载
1. 读取RGB图、深度图和碰撞图
2. 归一化深度图到[0, 255]范围
3. 添加噪声，增加数据增强（仅在不包含RGB时）
4. 返回处理后的深度图和碰撞图，归一化到[0, 1]范围（默认）
5. 可选返回RGB图像
"""


# 数据预处理函数
def preprocess_image(image):
    """将图像归一化到 [0.0, 1.0] 并转换为 PyTorch 张量"""
    image = image / 255.0
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def preprocess_rgb_image(image):
    """将RGB图像归一化到 [0.0, 1.0] 并转换为 PyTorch 张量"""
    image = image / 255.0
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW


# 数据集类
class RGBDepthCollisionDataset(Dataset):
    def __init__(self, depths_folder, colls_folder, rgbs_folder=None,
                 transform=preprocess_image, rgb_transform=preprocess_rgb_image,
                 return_file_name=False, include_rgb=False, is_simulate=False):
        self.depths_folder = depths_folder
        self.colls_folder = colls_folder
        self.rgbs_folder = rgbs_folder
        self.transform = transform
        self.rgb_transform = rgb_transform
        self.include_rgb = include_rgb
        self.is_simulate = is_simulate

        # 获取深度图和碰撞图的文件名列表
        self.depth_files = sorted([f for f in os.listdir(depths_folder) if f.endswith(".png")])
        self.coll_files = sorted([f for f in os.listdir(colls_folder) if f.endswith(".png")])

        # 如果包含RGB，则获取RGB文件列表
        if self.include_rgb:
            assert rgbs_folder is not None, "当include_rgb=True时，必须提供rgbs_folder"
            self.rgb_files = sorted([f for f in os.listdir(rgbs_folder) if f.endswith(".png")])
            assert len(self.rgb_files) == len(self.depth_files) == len(self.coll_files), \
                "RGB图、深度图和碰撞图的数量不一致"

            # 检查文件名是否匹配
            for rgb_file, depth_file, coll_file in zip(self.rgb_files, self.depth_files, self.coll_files):
                assert rgb_file == depth_file == coll_file, f"文件名不匹配: {rgb_file} vs {depth_file} vs {coll_file}"
        else:
            # 只检查深度图和碰撞图
            assert len(self.depth_files) == len(self.coll_files), "深度图和碰撞图的数量不一致"
            # for depth_file, coll_file in zip(self.depth_files, self.coll_files):
            #     assert depth_file == coll_file, f"文件名不匹配: {depth_file} vs {coll_file}"

        # 如果是仿真数据，过滤文件名
        if self.is_simulate:
            self.depth_files = [f for f in self.depth_files if f.startswith("depth_")]
            self.coll_files = [f for f in self.coll_files if f.startswith("depth_")]
            if self.include_rgb:
                self.rgb_files = [f for f in self.rgb_files if f.startswith("depth_")]

            # 确保过滤后文件数量一致
            assert len(self.depth_files) == len(self.coll_files), "过滤后深度图和碰撞图的数量不一致"
            if self.include_rgb:
                assert len(self.depth_files) == len(self.rgb_files), "过滤后RGB图和深度图的数量不一致"

        self.IMAGE_SIZE = (480, 270)  # 图像大小
        self.MIN_DEPTH = int(0.2 * 255 / 10)  # 5
        self.MAX_DEPTH = 255
        self.return_file_name = return_file_name

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
        noisy_depth_image = np.clip(depth_image + noise, 0, self.MAX_DEPTH)
        return noisy_depth_image.astype(np.uint8)

    def enhanced_augmentation(self, depth_image, coll_image):
        """增强的数据增强，包含更多样的噪声类型"""
        # 混合噪声：深度相关噪声 + 极端低深度噪声
        depth_image = self.add_mixed_noise(depth_image)
        return depth_image, coll_image

    def add_extreme_low_depth_noise(self, depth_image, base_probability=0.5):
        """
        专门添加接近0的深度噪声，主要在深度值较低的区域添加
        """
        noisy_depth = depth_image.astype(np.float32)

        # 创建基于深度的概率掩码
        probability_map = np.zeros_like(depth_image, dtype=np.float32)

        # 只在深度值小于等于2*MIN_DEPTH的区域添加噪声
        low_depth_region = depth_image <= (3 * self.MIN_DEPTH)

        # 在低深度区域内，深度值越低，添加噪声的概率越高
        if np.any(low_depth_region):
            low_depth_values = depth_image[low_depth_region].astype(np.float32)
            # 线性映射：深度为0时概率最高，深度为2*MIN_DEPTH时概率最低
            region_probabilities = base_probability * (1.0 - low_depth_values / (3 * self.MIN_DEPTH))
            probability_map[low_depth_region] = region_probabilities

        # 根据概率图生成噪声掩码
        low_depth_mask = np.random.rand(*depth_image.shape) < probability_map

        # 生成接近0的噪声值 [0, MIN_DEPTH*0.3]
        low_noise = np.random.uniform(0, self.MIN_DEPTH, depth_image.shape)
        noisy_depth[low_depth_mask] = low_noise[low_depth_mask]

        return np.clip(noisy_depth, 0, self.MAX_DEPTH).astype(np.uint8)

    def add_mixed_noise(self, depth_image):
        """混合噪声：深度相关噪声 + 极端低深度噪声"""
        # 先加深度相关噪声
        noisy_depth = self.add_depth_dependent_noise(depth_image)
        # 再加极端低深度噪声
        noisy_depth = self.add_extreme_low_depth_noise(noisy_depth.astype(np.uint8))
        return noisy_depth

    # 增加数据增强 主要是图像翻转
    def augment_image(self, depth, colls, rgb=None):
        """随机翻转图像"""
        flip_h = np.random.rand() < 0.5
        flip_v = np.random.rand() < 0.2

        if flip_h:
            depth = cv2.flip(depth, 1)  # 水平翻转
            colls = cv2.flip(colls, 1)  # 水平翻转
            if rgb is not None:
                rgb = cv2.flip(rgb, 1)  # 水平翻转
        if flip_v:
            depth = cv2.flip(depth, 0)  # 垂直翻转
            colls = cv2.flip(colls, 0)  # 垂直翻转
            if rgb is not None:
                rgb = cv2.flip(rgb, 0)  # 垂直翻转

        if rgb is not None:
            return rgb, depth, colls
        return depth, colls

    def __getitem__(self, index):
        depth_path = os.path.join(self.depths_folder, self.depth_files[index])
        coll_path = os.path.join(self.colls_folder, self.coll_files[index])

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        coll_image = cv2.imread(coll_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        depth_image = self.uint8_min_normalize(depth_image)

        # 使用增强的数据增强（包含低深度噪声）- 仅在不需要RGB时添加噪声
        if not self.include_rgb and np.random.rand() < 0.7:  # 只在非RGB模式下添加噪声
            depth_image, coll_image = self.enhanced_augmentation(depth_image, coll_image)

        # 归一化深度图
        depth_image = self.uint8_0_normalize(depth_image)

        # 如果包含RGB，读取并处理RGB图像
        if self.include_rgb:
            rgb_path = os.path.join(self.rgbs_folder, self.rgb_files[index])
            rgb_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # 转换为RGB
            rgb_image = cv2.resize(rgb_image, self.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

            # 数据增强（几何变换）
            rgb_image, depth_image, coll_image = self.augment_image(depth_image, coll_image, rgb_image)

            # 应用转换为tensor
            if self.rgb_transform:
                rgb_image = self.rgb_transform(rgb_image)
        else:
            # 数据增强（几何变换）
            depth_image, coll_image = self.augment_image(depth_image, coll_image)

        # 应用转换为tensor
        if self.transform:
            depth_image = self.transform(depth_image)
            coll_image = self.transform(coll_image)

        if self.return_file_name:
            if self.include_rgb:
                return rgb_image, depth_image, coll_image, self.depth_files[index]
            else:
                return depth_image, coll_image, self.depth_files[index]
        else:
            if self.include_rgb:
                return rgb_image, depth_image, coll_image
            else:
                return depth_image, coll_image

    def uint8_0_normalize(self, depth_map):
        """
        将深度图归一化到[0, 255]范围，大于MAX_DEPTH设置为255 小于MIN_DEPTH的像素值设置为0
        返回值范围[0, 255]的深度图
        """
        # depth_map[(depth_map <= self.MIN_DEPTH) & (depth_map > 0)] = 0
        return depth_map

    def uint8_min_normalize(self, normalized_depth_map):
        """保留[0, min_depth]范围内的信息"""
        max_val = np.max(normalized_depth_map)
        if max_val > 255:
            normalized_depth_map = normalized_depth_map * 255 / 7000

        # 不再将小值截断到MIN_DEPTH，而是保留原始相对关系
        normalized_depth_map = np.clip(normalized_depth_map, 0, self.MAX_DEPTH)
        normalized_depth_map = cv2.resize(normalized_depth_map, self.IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        return normalized_depth_map.astype(np.uint8)


if __name__ == "__main__":
    # 测试函数
    def test_dataset(dataloader, include_rgb=False):
        """测试数据集和数据加载器"""

        # 测试数据加载器
        print(f"\n测试数据加载器 (include_rgb={include_rgb}):")
        for batch in dataloader:
            if include_rgb:
                rgb_batch, depth_batch, coll_batch, filenames = batch
                num_samples = min(3, len(filenames))
                plt.figure(figsize=(15, 3 * num_samples))

                for i in range(num_samples):
                    # RGB图像
                    plt.subplot(num_samples, 3, 3 * i + 1)
                    rgb_img = rgb_batch[i].permute(1, 2, 0).numpy()  # CHW to HWC
                    plt.imshow(rgb_img)
                    plt.title(f"RGB - {filenames[i]}")
                    plt.axis('off')

                    # 深度图
                    plt.subplot(num_samples, 3, 3 * i + 2)
                    plt.imshow(depth_batch[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                    plt.title(f"Clean Depth - {filenames[i]}")
                    plt.axis('off')

                    # 碰撞图
                    plt.subplot(num_samples, 3, 3 * i + 3)
                    plt.imshow(coll_batch[i].squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
                    plt.title(f"Collision - {filenames[i]}")
                    plt.axis('off')
            else:
                depth_batch, coll_batch, filenames = batch
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
            break  # 只显示一个批次


    # 数据集路径
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"
    rgbs_folder = "/home/niu/workspaces/VAE_ws/datasets/rgbs"

    # 测试不包含RGB的情况（与之前兼容）
    print("=== 测试不包含RGB的情况 ===")
    dataset_no_rgb = RGBDepthCollisionDataset(
        depths_folder=depths_folder,
        colls_folder=colls_folder,
        transform=preprocess_image,
        return_file_name=True,
        include_rgb=False,  # 默认值，可以省略
        is_simulate=False  # 默认值，可以省略
    )

    dataloader_no_rgb = DataLoader(
        dataset_no_rgb,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    test_dataset(dataloader_no_rgb, include_rgb=False)

    # 测试包含RGB的情况
    print("\n=== 测试包含RGB的情况 ===")
    dataset_with_rgb = RGBDepthCollisionDataset(
        depths_folder=depths_folder,
        colls_folder=colls_folder,
        rgbs_folder=rgbs_folder,
        transform=preprocess_image,
        rgb_transform=preprocess_rgb_image,
        return_file_name=True,
        include_rgb=True,
        is_simulate=False  # 默认值，可以省略
    )

    dataloader_with_rgb = DataLoader(
        dataset_with_rgb,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    test_dataset(dataloader_with_rgb, include_rgb=True)

    # 测试包含RGB的情况
    print("\n=== 测试不包含真实数据集的情况 ===")
    dataset_simulate = RGBDepthCollisionDataset(
        depths_folder=depths_folder,
        colls_folder=colls_folder,
        rgbs_folder=rgbs_folder,
        transform=preprocess_image,
        rgb_transform=preprocess_rgb_image,
        return_file_name=True,
        include_rgb=True,
        is_simulate=True  # 默认值，可以省略
    )

    dataloader_simulate = DataLoader(
        dataset_simulate,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    test_dataset(dataloader_simulate, include_rgb=True)