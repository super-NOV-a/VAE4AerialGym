import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt



# 全局常量配置
IMAGE_SIZE = (480, 270)
CAMERA_HFOV_DEG = 87
DRONE_SIZE_METERS = 0.5  # 无人机直径0.5米
MAX_DILATION_SIZE = 50
MIN_DEPTH = int(0.2 * 255 / 10)  # 5
MAX_DEPTH = 255

ASPECT_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]
HFOV_RAD = np.radians(CAMERA_HFOV_DEG)
VFOV_RAD = 2 * np.arctan(np.tan(HFOV_RAD / 2) / ASPECT_RATIO)

fx = IMAGE_SIZE[0] / (2 * np.tan(HFOV_RAD / 2))
fy = IMAGE_SIZE[1] / (2 * np.tan(VFOV_RAD / 2))
cx = IMAGE_SIZE[0] / 2
cy = IMAGE_SIZE[1] / 2

CAMERA_MATRIX = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])


def calculate_dilation_size(depth_value):
    """根据深度值计算膨胀大小"""
    if depth_value <= 0:
        return 1

    # 将深度图值转换为实际距离（米）
    depth_in_meters = (depth_value / MAX_DEPTH) * 10.0

    # 计算无人机在当前深度下的图像平面投影大小（半径）
    half_drone_size = DRONE_SIZE_METERS / 2  # 0.25米半径

    # 根据相似三角形原理计算图像平面中的半径（像素）
    dilation_radius_x = (half_drone_size * fx) / depth_in_meters
    dilation_radius_y = (half_drone_size * fy) / depth_in_meters

    # 取较大的膨胀半径并转换为整数
    dilation_radius = int(max(dilation_radius_x, dilation_radius_y))

    # 限制在合理范围内
    return max(min(dilation_radius, MAX_DILATION_SIZE), 1)


def pixel_wise_dilation_optimized(depth_map, max_dilation=MAX_DILATION_SIZE):
    """
    逐像素膨胀深度图，考虑所有像素点，并按深度值排序以避免远处覆盖近处。

    参数：
        depth_map (numpy.ndarray): 输入深度图，值范围 [0, 255]
        max_dilation (int): 最大膨胀半径

    返回：
        dilated_depth_map (numpy.ndarray): 膨胀后的深度图
    """
    # 初始化膨胀后的深度图
    dilated_depth_map = np.full_like(depth_map, 255)

    # 获取所有有效像素点（深度值非255且大于5）
    valid_pixels = np.where((depth_map != 255) & (depth_map > 5))
    if valid_pixels[0].size == 0:
        dilated_depth_map[dilated_depth_map == 255] = 0
        return dilated_depth_map  # 如果没有有效像素，直接返回

    depth_values = depth_map[valid_pixels]
    y_coords, x_coords = valid_pixels

    # 计算每个像素点的膨胀半径
    dilation_sizes = np.array([calculate_dilation_size(d) for d in depth_values], dtype=np.int32)
    dilation_sizes = np.clip(dilation_sizes, 1, max_dilation)

    # 按深度值降序排序（确保较远的像素先处理）
    sort_indices = np.argsort(depth_values)[::-1]
    depth_values_sorted = depth_values[sort_indices]
    y_coords_sorted = y_coords[sort_indices]
    x_coords_sorted = x_coords[sort_indices]
    dilation_sizes_sorted = dilation_sizes[sort_indices]

    # 处理每个像素
    for y, x, d_size, depth_val in zip(y_coords_sorted, x_coords_sorted, dilation_sizes_sorted, depth_values_sorted):
        # 计算膨胀区域的边界
        min_y = max(0, y - d_size)
        max_y = min(depth_map.shape[0], y + d_size + 1)
        min_x = max(0, x - d_size)
        max_x = min(depth_map.shape[1], x + d_size + 1)

        # 获取区域内的当前深度值
        region = dilated_depth_map[min_y:max_y, min_x:max_x]

        # 更新区域内的深度值，保留较小的深度值
        updated_region = np.where(region > depth_val, depth_val, region)
        dilated_depth_map[min_y:max_y, min_x:max_x] = updated_region

    # 将未处理的区域设置为0
    dilated_depth_map[dilated_depth_map == 255] = 0

    return dilated_depth_map


def custom_normalize(normalized_depth_map):
    """
    将深度图归一化到[0, 255]范围，(大于0)&(小于MIN_DEPTH)的像素值设置为MIN_DEPTH
    返回值范围[0, 255]的深度图
    """
    max_val = np.max(normalized_depth_map)
    if max_val > 255:
        normalized_depth_map = normalized_depth_map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
    else:
        normalized_depth_map = normalized_depth_map
    normalized_depth_map[(normalized_depth_map < MIN_DEPTH) & (normalized_depth_map > 0)] = MIN_DEPTH
    normalized_depth_map = cv2.resize(normalized_depth_map, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return normalized_depth_map.astype(np.uint8)


def process_depth_image(depth_file_path):
    # 读取深度图
    depth_map = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise ValueError(f"无法加载深度图文件：{depth_file_path}")

    # 归一化到0-255范围
    depth_map = custom_normalize(depth_map.astype(np.float32))
    # 对深度值减去0.25米后重新计算  # 此处进行了原始深度的偏置 表示无人机实际发生碰撞的位置！！！！！！！！！！！！！！！！
    range_map = (depth_map * 10.0 / 255) - 0.25
    range_map = np.clip(range_map, 0.2, 10.0)
    new_depth_map = ((range_map / 10.0) * 255).astype(np.uint8)

    # 应用逐像素膨胀
    dilated_depth_map = pixel_wise_dilation_optimized(new_depth_map)

    return depth_map, dilated_depth_map


def plot_results(original_depth, dilated_depth):
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.imshow(original_depth, vmin=0, vmax=255)
    plt.title("Original Depth Map")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dilated_depth, vmin=0, vmax=255)
    plt.title("Dilated Depth Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

"""
该代码用于处理深度图像，主要包括以下功能：
1. 读取深度图像并进行预处理。
2. 根据深度值计算每个像素的膨胀大小。
3. 逐像素膨胀深度图像，避免远处的像素覆盖近处的像素。
4. 将处理后的深度图像进行归一化，并调整大小。
5. 可视化原始深度图像和膨胀后的深度图像。
6. 计算膨胀大小的函数。
7. 处理深度图像的主函数。

"""

if __name__ == "__main__":
    start_time = time.time()  # 开始计时
    # depth_file = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_15339.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/raw_png/in_k_00_160120_000001_rd.png" # 不work
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/up_png/in_k_00_160120_000002_ud.png"
    depth_file = "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_11.png"

    original_depth, dilated_depth = process_depth_image(depth_file)
    plot_results(original_depth, dilated_depth)
    end_time = time.time()  # 结束计时
    print(f"总耗时：{end_time - start_time} 秒")

"""
最原始的逐像素膨胀  对于某些存在无效像素（小于5）的位置有可能发生远处的像素覆盖近处的像素，如anomaly_11.png
"""

