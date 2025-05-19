# 文件 1: ICRA_dataset.py
import math
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

"""
该文件使用的是ICRA论文的操作方法，提取边缘后进行膨胀处理
1. 读取深度图
2. 归一化深度图到[0, 255]范围
3. 填充深度图中的零值
4. 生成边缘图
5. 计算边缘上每个像素的膨胀大小
6. 逐像素膨胀边缘
7. 从边缘和原始深度图生成碰撞图
8. 返回处理后的碰撞图
"""

# 设置随机数种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 对所有GPU设置相同的随机种子
torch.backends.cudnn.deterministic = True  # 确保CUDA使用确定性算法
torch.backends.cudnn.benchmark = False  # 禁用CuDNN自动寻找最合适的算法


# ==================== 统一常量管理 ====================
class Config:
    # 图像参数
    IMAGE_SIZE = (480, 270)
    RESIZE_SHAPE = (480, 270)

    # 相机参数
    CAMERA_HFOV_DEG = 87
    MAX_DILATION_SIZE = 100
    CAMERA_MATRIX = None

    # 无人机参数
    DRONE_DIAMETER = 0.25  # 无人机直径（米）
    DRONE_HEIGHT = 0.25  # 无人机高度（米）

    # 深度参数
    DEPTH_RANGE = 10.0  # 米
    MIN_DEPTH = int(0.2 * 255 / 10)
    MAX_DEPTH = 255

    @classmethod
    def init_camera_matrix(cls):
        aspect_ratio = cls.IMAGE_SIZE[0] / cls.IMAGE_SIZE[1]
        hfov_rad = np.radians(cls.CAMERA_HFOV_DEG)
        vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) / aspect_ratio)

        fx = cls.IMAGE_SIZE[0] / (2 * np.tan(hfov_rad / 2))
        fy = cls.IMAGE_SIZE[1] / (2 * np.tan(vfov_rad / 2))

        cls.CAMERA_MATRIX = np.array([
            [fx, 0, cls.IMAGE_SIZE[0] / 2],
            [0, fy, cls.IMAGE_SIZE[1] / 2],
            [0, 0, 1]
        ])

# 初始化相机参数
Config.init_camera_matrix()

# ==================== 膨胀核缓存 ====================


class DilationKernelCache:
    _cache = {}

    @classmethod
    def get_kernel(cls, dy, dx):
        key = (dy, dx)
        if key not in cls._cache:
            cls._cache[key] = np.ones((2 * dy + 1, 2 * dx + 1), dtype=np.uint8)
        return cls._cache[key]


# ==================== 数据集类 ====================
class DepthCollideDataset(Dataset):
    def __init__(self, depths_folder, transform=None, augment=False):
        super().__init__()
        self.depths_folder = depths_folder
        self.depth_files = sorted([f for f in os.listdir(depths_folder) if f.endswith(".png")])
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.depth_files)

    def __getitem__(self, index):
        # 加载深度图
        depth_path = os.path.join(self.depths_folder, self.depth_files[index])
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        # 预处理
        filled_depth = self.preprocess_depth(depth_image)

        # 生成碰撞图
        edge_depth = self.generate_edges(filled_depth)
        coll_map = self.generate_collision_map(filled_depth, edge_depth)

        # 数据增强
        if self.augment and random.random() < 0.5:
            filled_depth = cv2.flip(filled_depth, 1)
            coll_map = cv2.flip(coll_map, 1)

        # 转换为Tensor
        if self.transform:
            filled_depth = self.transform(filled_depth)
            coll_map = self.transform(coll_map)

        return filled_depth, coll_map

    def preprocess_depth(self, raw_depth):
        """统一预处理流程"""
        # 归一化
        normalized = self.custom_normalize(raw_depth)
        # 填充
        return self.inpaint_zeros(normalized)

    def custom_normalize(self, depth_map):
        """深度图归一化"""
        if np.max(depth_map) > 255:
            normalized = depth_map * 255 / 7000
        else:
            normalized = depth_map.copy()
        normalized = np.clip(normalized, Config.MIN_DEPTH, 255)
        return cv2.resize(normalized.astype(np.uint8), Config.RESIZE_SHAPE)

    def inpaint_zeros(self, depth_map):
        """零值填充"""
        zero_mask = (depth_map <= 0).astype(np.uint8)
        return cv2.inpaint(depth_map, zero_mask, 3, cv2.INPAINT_TELEA)

    def generate_edges(self, depth_map):
        """边缘检测(忽略深度≤5的像素)"""
        edges = cv2.Canny(depth_map, 30, 100)    # 生成原始边缘
        ignore_mask = (depth_map <= 5)  # 创建深度掩膜(True表示需要忽略的位置)
        edges_cleaned = np.where(ignore_mask, 0, edges)     # 在边缘图中清除需要忽略的区域
        edge_depth = np.full_like(depth_map, 255)   # 生成带深度值的边缘图
        edge_depth[edges_cleaned != 0] = depth_map[edges_cleaned != 0]
        return edge_depth

    def generate_collision_map(self, orig_depth, edge_depth):
        """生成碰撞图"""
        dilated = self.dilate_edges(edge_depth)
        coll_map = np.zeros_like(dilated)
        none_zeros = (dilated != 0)
        zeros = (dilated == 0)
        coll_map[zeros] = orig_depth[zeros]
        coll_map[none_zeros] = dilated[none_zeros]
        return coll_map # cv2.dilate(coll_map, np.ones((3, 3), np.uint8))

    def dilate_edges(self, edge_depth):
        """优化后的膨胀处理"""
        dilated = np.full_like(edge_depth, 255)
        y_coords, x_coords = np.where(edge_depth != 255)

        # 批量计算膨胀参数
        depths = edge_depth[y_coords, x_coords]
        params = [self.calculate_dilation(d) for d in depths]

        # 向量化处理
        for idx in range(len(y_coords)):
            y, x, (dy, dx) = y_coords[idx], x_coords[idx], params[idx]
            self.apply_dilation(dilated, y, x, dy, dx, depths[idx])

        dilated[dilated == 255] = 0
        return cv2.dilate(dilated, np.ones((3, 3), np.uint8))

    def calculate_dilation(self, depth_val):
        """计算膨胀尺寸"""
        depth_m = (depth_val / 255.0) * Config.DEPTH_RANGE
        depth_m = max(depth_m, 0.2)

        fx = Config.CAMERA_MATRIX[0, 0]
        fy = Config.CAMERA_MATRIX[1, 1]

        dx = int(fx * (Config.DRONE_DIAMETER / depth_m))
        dy = int(fy * (Config.DRONE_HEIGHT / depth_m))

        dx = np.clip(dx, 1, Config.MAX_DILATION_SIZE)
        dy = np.clip(dy, 1, Config.MAX_DILATION_SIZE)
        return (dy, dx)

    def apply_dilation(self, target, y, x, dy, dx, depth_val):
        """应用单个膨胀操作"""
        y_start = max(0, y - dy)
        y_end = min(target.shape[0], y + dy + 1)
        x_start = max(0, x - dx)
        x_end = min(target.shape[1], x + dx + 1)

        kernel = DilationKernelCache.get_kernel(dy, dx)
        cropped_kernel = kernel[:y_end - y_start, :x_end - x_start]

        roi = target[y_start:y_end, x_start:x_end]
        mask = (cropped_kernel == 1) & ((roi > depth_val) | (roi == 255))
        roi[mask] = depth_val


# ==================== 其余辅助函数 ====================
def preprocess_image(image):
    """图像归一化"""
    return torch.tensor(image / 255.0, dtype=torch.float32).unsqueeze(0)


if __name__ == "__main__":

    def test_dataloader(dataloader, device):
        # 测试数据集
        for depth_batch, coll_batch in dataloader:
            coll_batch = coll_batch.to(device)
            depth_batch = depth_batch.to(device)
            # 可视化
            depth_batch = depth_batch.cpu().numpy()
            coll_batch = coll_batch.cpu().numpy()

            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            for i in range(5):
                axes[0, i].imshow(depth_batch[i, 0], vmin=0, vmax=1)
                axes[0, i].set_title("Original Depth")
                axes[0, i].axis('off')

                axes[1, i].imshow(coll_batch[i, 0], vmin=0, vmax=1)
                axes[1, i].set_title("Original Depth Collision")
                axes[1, i].axis('off')

            plt.tight_layout()
            plt.show()
            # return depth_batch, coll_batch

    # 数据集路径
    depths_folder = "/home/niu/下载/depth_images" # "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/datasets/depths"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备数据集
    dataset = DepthCollideDataset(depths_folder, transform=preprocess_image, augment=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 测试模型
    test_dataloader(dataloader, device)

