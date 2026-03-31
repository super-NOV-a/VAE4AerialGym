import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# 全局常量配置
DRONE_SIZE_METERS = 0.5  # 无人机直径0.5米
DRONE_HALF_SIZE_METERS = DRONE_SIZE_METERS / 2  # 无人机半径
MAX_DILATION_SIZE = 40

MIN_DEPTH = int(0.2 * 255 / 10) # 5
MAX_DEPTH = 255

IMAGE_SIZE = (480, 270)
CAMERA_HFOV_DEG = 87
ASPECT_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]  # ≈ 1.7778
HFOV_RAD = np.radians(CAMERA_HFOV_DEG)
VFOV_RAD = 2 * np.arctan(np.tan(HFOV_RAD/2) / ASPECT_RATIO)
# vfov_deg = np.degrees(VFOV_RAD)  # ≈ 56.1°

fx = IMAGE_SIZE[0] / (2 * np.tan(HFOV_RAD/2))  # ≈ 252.9（假设无畸变）
fy = IMAGE_SIZE[1] / (2 * np.tan(VFOV_RAD/2)) # ≈ 252.9

CAMERA_MATRIX = np.array([
    [fx, 0,  IMAGE_SIZE[0]/2],
    [0,  fy, IMAGE_SIZE[1]/2],
    [0,  0,  1 ]
])

def preprocess_image(depth_image):
    """
    深度图预处理:
    1. 归一化到[0, 255]范围
    2. 应用深度范围约束
    3. 调整尺寸
    4. 转换为模型输入张量

    参数:
        depth_image (np.ndarray): 输入深度图

    返回:
        torch.Tensor: 预处理后的张量
    """
    # 归一化处理
    if np.max(depth_image) > 255:
        depth_image = depth_image * 255 / 7000  # DIML/CVl RGB-D数据集特定缩放

    # 应用深度约束
    depth_image[depth_image > MAX_DEPTH] = MAX_DEPTH
    depth_image[depth_image < MIN_DEPTH] = 0

    # 调整尺寸
    depth_image = cv2.resize(depth_image, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return depth_image

def uint8_normalize(depth_map):
    """
    将深度图归一化到[0, 255]范围，(大于0)&(小于MIN_DEPTH)的像素值设置为MIN_DEPTH
    返回值范围[0, 255]的深度图
    """
    max_val = np.max(depth_map)
    if max_val > 255:
        normalized_depth_map = depth_map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
    else:
        normalized_depth_map = depth_map.copy()
    normalized_depth_map[normalized_depth_map > MAX_DEPTH] = MAX_DEPTH
    normalized_depth_map[normalized_depth_map < MIN_DEPTH] = 0
    return normalized_depth_map.astype(np.uint8)

def apply_drone_offset(original_map, to255=True):
    """
    应用无人机尺寸偏移
    输入范围应该是 [0, 255] 的深度图像
    输出范围是 [5, 255] 的深度图像
    """
    depth_float = original_map.astype(np.float32) / MAX_DEPTH * 10.0  # 转换为米
    offset_depth = depth_float - DRONE_HALF_SIZE_METERS  # 减去无人机尺寸
    offset_depth[offset_depth < 0] = 0  # 处理负值
    if to255:
        offset_depth = (offset_depth / 10.0 * MAX_DEPTH).astype(np.uint8)
        offset_depth[offset_depth < 5] = 5
    else:
        offset_depth[offset_depth < 0.2] = 0.2  # 范围是[0.2, 10.0]米
    return offset_depth