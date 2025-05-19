import math
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 全局常量配置
IMAGE_SIZE = (480, 270)
CAMERA_HFOV_DEG = 87
DRONE_SIZE_METERS = 0.5  # 无人机直径0.5米
DRONE_HALF_SIZE_METERS = DRONE_SIZE_METERS / 2  # 无人机半径
MAX_DILATION_SIZE = 50

MIN_DEPTH = int(0.2 * 255 / 10)  # 5
MAX_DEPTH = 255

ASPECT_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]  # ≈ 1.7778
HFOV_RAD = np.radians(CAMERA_HFOV_DEG)
VFOV_RAD = 2 * np.arctan(np.tan(HFOV_RAD / 2) / ASPECT_RATIO)
# vfov_deg = np.degrees(VFOV_RAD)  # ≈ 56.1°

fx = IMAGE_SIZE[0] / (2 * np.tan(HFOV_RAD / 2))  # ≈ 252.9（假设无畸变）
fy = IMAGE_SIZE[1] / (2 * np.tan(VFOV_RAD / 2))  # ≈ 252.9

CAMERA_MATRIX = np.array([
    [fx, 0, IMAGE_SIZE[0] / 2],
    [0, fy, IMAGE_SIZE[1] / 2],
    [0, 0, 1]
])


def _plot_preprocess_results(original, mask, resized):
    """可视化预处理结果"""
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(custom_normalize(original), cmap="gray", vmin=0, vmax=255)
    plt.title("Original Depth Map")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Zero Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(resized, cmap="gray", vmin=0, vmax=255)
    plt.title("Resized Depth Map")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def custom_normalize(map):
    """
    将深度图归一化到[0, 255]范围，(大于0)&(小于MIN_DEPTH)的像素值设置为MIN_DEPTH
    返回值范围[0, 255]的深度图
    """
    max_val = np.max(map)
    if max_val > 255:
        normalized_depth_map = map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
    else:
        normalized_depth_map = map.copy()
    normalized_depth_map[normalized_depth_map > MAX_DEPTH] = MAX_DEPTH
    normalized_depth_map[normalized_depth_map < MIN_DEPTH] = 0
    return normalized_depth_map.astype(np.uint8)


def resize_and_fill_depth(depth_map, half_plot=False):
    """
    调整深度图大小并填充零值区域，小于等于MIN_DEPTH的像素值加入零值掩膜
    """
    depth_map_copy = depth_map.copy()
    # 其他部分保持不变
    uint8_depth = custom_normalize(depth_map_copy)

    # 将小于等于0的像素值加入零值掩膜
    zero_mask = (uint8_depth <= MIN_DEPTH).astype(np.uint8)

    uint8_depth = apply_drone_offset(uint8_depth)
    uint8_depth_fill = cv2.inpaint(uint8_depth, zero_mask, 3, cv2.INPAINT_TELEA)
    uint8_depth_fill = cv2.resize(uint8_depth_fill, (480, 270), interpolation=cv2.INTER_LINEAR)
    zero_mask_resized = cv2.resize(zero_mask, (480, 270), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    uint8_depth = cv2.resize(uint8_depth, (480, 270), interpolation=cv2.INTER_LINEAR)

    if half_plot and zero_mask_resized is not None:
        _plot_preprocess_results(depth_map, zero_mask_resized, uint8_depth)
    return zero_mask_resized, uint8_depth, uint8_depth_fill


def dilate_zero_mask(zero_mask, dilation_iterations=1):
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(zero_mask, kernel, iterations=dilation_iterations)
    return dilated_mask


# def generate_edges_from_depth(depth_map, zero_mask, edge_threshold_low=30, edge_threshold_high=50):
#     edges = cv2.Canny(depth_map, edge_threshold_low, edge_threshold_high)
#
#     edge_depth = depth_map.copy()
#     edge_depth[:, :] = 255
#     edge_depth[edges != 0] = depth_map[edges != 0]  # 边缘不为255的区域保留原始深度
#
#     # # # 再次使用zero_mask过滤掉零值区域的边缘
#     # edge_depth[zero_mask == 1] = 255
#     return depth_map, edge_depth

def generate_edges_from_depth(depth_map, zero_mask, edge_threshold_low=30, edge_threshold_high=50):
    # 对zero_mask进行膨胀操作，扩大零值区域
    dilated_zero_mask = dilate_zero_mask(zero_mask, dilation_iterations=1)

    # 将膨胀后的零值掩膜应用于深度图，填充零值区域边界
    depth_map_filled = depth_map.copy()
    depth_map_filled[dilated_zero_mask == 1] = 255  # 或者使用周围像素的平均值进行填充

    hist_eq_depth = cv2.equalizeHist(depth_map_filled)
    normalized_depth = cv2.normalize(hist_eq_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(normalized_depth, edge_threshold_low, edge_threshold_high)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_depth_map = cv2.erode(depth_map_filled, kernel, iterations=1)

    edge_depth = depth_map_filled.copy()
    edge_depth[:, :] = 255
    edge_depth[edges != 0] = filtered_depth_map[edges != 0]

    # 再次使用zero_mask过滤掉零值区域的边缘
    edge_depth[zero_mask == 1] = 255

    return depth_map, edge_depth


def calculate_dilation_size(depth_value):
    """根据深度值计算膨胀大小"""
    if depth_value <= 0:
        return 1

    # 将深度图值转换为实际距离（米）
    depth_in_meters = (depth_value / MAX_DEPTH) * 10.0

    # 根据相似三角形原理计算图像平面中的半径（像素）
    dilation_radius_x = (DRONE_HALF_SIZE_METERS * fx) / depth_in_meters
    dilation_radius_y = (DRONE_HALF_SIZE_METERS * fy) / depth_in_meters

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
    # 初始化膨胀后的深度图，使用0填充
    dilated_depth_map = np.full_like(depth_map, 0)

    # 获取所有有效像素点（深度值非0才是有效的）
    valid_pixels = np.where((depth_map != 0))
    if valid_pixels[0].size == 0:
        # 如果没有有效像素，将所有像素设置为0
        dilated_depth_map[:] = 0
        return dilated_depth_map

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

        # 创建掩膜：仅更新为0的值或者当前区域值大于深度值的位置
        mask = (region == 0) | (region > depth_val)
        # 更新区域内的深度值，保留较小的深度值
        updated_region = np.where(mask, depth_val, region)
        dilated_depth_map[min_y:max_y, min_x:max_x] = updated_region

    # # dilated_depth_map[dilated_depth_map <= MIN_DEPTH & dilated_depth_map > 0] = MIN_DEPTH
    # dilated_depth_map[0 < dilated_depth_map < MIN_DEPTH] = MIN_DEPTH
    dilated_depth_map = np.where(dilated_depth_map > MIN_DEPTH, dilated_depth_map,
                                 MIN_DEPTH * np.ones_like(depth_map))
    return dilated_depth_map.astype(np.uint8)


def edge_dilation_optimized(edge_depth, max_dilation=MAX_DILATION_SIZE):
    dilated_edge_depth = np.full_like(edge_depth, 255)
    edge_pixels = np.where((edge_depth != 255)&(edge_depth > 5))    # 膨胀的像素点需要满足有效像素值
    depth_values = edge_depth[edge_pixels]
    y_coords, x_coords = edge_pixels
    dilation_sizes = np.array([calculate_dilation_size(d) for d in depth_values], dtype=np.int32)
    dilation_sizes = np.clip(dilation_sizes, 1, max_dilation)

    for y, x, d_size in zip(y_coords, x_coords, dilation_sizes):
        min_y = max(0, y - d_size)
        max_y = min(edge_depth.shape[0], y + d_size)
        min_x = max(0, x - d_size)
        max_x = min(edge_depth.shape[1], x + d_size)
        dilated_edge_depth[min_y:max_y, min_x:max_x] = np.minimum(
            dilated_edge_depth[min_y:max_y, min_x:max_x],
            edge_depth[y, x]
        )
    dilated_edge_depth = cv2.dilate(
        dilated_edge_depth,
        np.ones((3, 3), np.uint8))
    return dilated_edge_depth


def generate_coll(depth_map1, depth_map2):
    # 创建一个与深度图大小相同的数组来存储结果
    result = np.zeros_like(depth_map1)
    # 找到两个深度图中非零值的索引
    non_zero_in_both = (depth_map1 != 0) & (depth_map2 != 0)
    non_zero_in_resized = (depth_map1 != 0) & (depth_map2 == 0)
    non_zero_in_dilated_edges = (depth_map1 == 0) & (depth_map2 != 0)

    # 对于只有深度图1有非零值的像素点，保留该值
    result[non_zero_in_resized] = depth_map1[non_zero_in_resized]
    # 对于只有深度图2有非零值的像素点，保留该值
    result[non_zero_in_dilated_edges] = depth_map2[non_zero_in_dilated_edges]
    # 对于两个深度图中都有非零值的像素点，保留较小的值
    result[non_zero_in_both] = np.minimum(depth_map1[non_zero_in_both], depth_map2[non_zero_in_both])
    # result[non_zero_in_both] = np.maximum(depth_map1[non_zero_in_both], depth_map2[non_zero_in_both])

    return result


def process_depth_pipeline(depth_file_path):
    """深度处理流水线封装函数
    参数：
        depth_file_path: 深度图文件路径
    返回：
        depth_map_resized: 预处理后的缩放深度图
        collisions: 最终碰撞检测结果图
    """
    # 读取深度图
    oridepth_map = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
    if oridepth_map is None:
        raise ValueError(f"无法加载深度图文件：{depth_file_path}")
    oridepth_map = oridepth_map.astype(np.float32)

    # # 确保深度图中的值在 [0, MAX_DEPTH] 范围内
    # oridepth_map = np.clip(oridepth_map, 0, MAX_DEPTH)

    # 预处理阶段
    zero_mask, uint8_depth_resized, uint8_depth_filled = resize_and_fill_depth(oridepth_map, False)

    # 边缘生成阶段
    _, edge_depth = generate_edges_from_depth(uint8_depth_resized, zero_mask)
    dilated_edges = edge_dilation_optimized(edge_depth)

    # # 将深度图值转换为实际距离（米）,然后考虑到碰撞实际位置的偏置后 从距离转换回深度
    # range_map = (uint8_depth_fill * 10.0 / 255) - DRONE_HALF_SIZE_METERS
    # range_map = np.clip(range_map, 0, 10.0)     # 小于0.2m表示无效距离
    # uint8_depth_fill = ((range_map / 10.0) * 255).astype(np.uint8)
    # 对fill图做一个逐点膨胀呢
    uint8_depth_fill = pixel_wise_dilation_optimized(uint8_depth_filled)

    # 后处理阶段
    collisions = generate_coll(uint8_depth_fill, dilated_edges)
    # collisions = cv2.dilate(collisions, np.ones((2, 2), np.uint8), iterations=1)
    # collisions = apply_drone_offset(collisions)

    d1 = uint8_depth_resized.astype(np.float32)
    d2 = collisions.astype(np.float32)

    # 计算绝对差异（处理负值）
    diff = np.abs(d1 - d2)
    diff[diff < 0] = 0
    # 归一化处理
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 可视化所有结果
    plot_results(uint8_depth_resized, zero_mask, uint8_depth_filled, uint8_depth_fill,
                 edge_depth, dilated_edges, collisions, diff_norm)

    return uint8_depth_resized, collisions


def plot_results(depth_resized, zero_mask, depth_map_resized, filled_depth_map, edge_depth,
                 dilated_edges, collisions, diff):
    plt.figure(figsize=(25, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(custom_normalize(depth_resized), vmin=0, vmax=255)
    plt.title("Resized Original Depth")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(zero_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Zero Mask")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(depth_map_resized, vmin=0, vmax=255)
    plt.title("Filled Depth")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(filled_depth_map, vmin=0, vmax=255)
    plt.title("Collision Filled Depth")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(edge_depth, cmap="gray_r")
    plt.title("Edge Depth")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(dilated_edges, vmin=0, vmax=255)
    plt.title("Dilated Edges")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(collisions, vmin=0, vmax=255)
    plt.title("Collisions")
    plt.axis("off")

    plt.subplot(2, 4, 8)
    plt.imshow(diff, vmin=0, vmax=255)
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def apply_drone_offset(original_map):
    """应用无人机尺寸偏移"""
    depth_float = original_map.astype(np.float32) / MAX_DEPTH * 10.0  # 转换为米
    offset_depth = depth_float - DRONE_HALF_SIZE_METERS  # 减去无人机尺寸
    offset_depth[offset_depth < 0] = 0  # 处理负值
    offset_depth_map = (offset_depth / 10.0 * MAX_DEPTH).astype(np.uint8)  # 转回深度图范围
    return offset_depth_map


"""
该文件实现了一个从深度图像生成碰撞图像的处理流水线。
该流水线包括以下步骤：
1. 读取深度图像（图1 深度图）
2. 预处理深度图像，包括调整大小和填充零值区域（图2 零值mask）和填充深度图像（图3 填充深度图）
4. 对填充后的深度图像进行膨胀（图4 填充后膨胀图）
3. 从原始深度图中生成边缘图像（图5 边缘图像）
4. 逐像素膨胀边缘图像（图6 膨胀边缘）
5. 生成碰撞图像（图7 碰撞=填充膨胀图+膨胀边缘）
6. 绘制碰撞图与原始深度图像的差异（图8 差异）
注意此处没有实现 深度偏移  ！！！！！todo
由于对零值的填充，导致有些无效像素的值被覆盖，可能会导致远处的像素覆盖近处的像素。没有无效像素时可以使用该方法中的绘图来作为展示
"""
if __name__ == "__main__":
    # 示例文件路径

    # depth_file = "/home/niu/下载/indoor_train-004/train/HR/02. Cafe/depth_vi/in_00_160315_165831_depth_vi.png"
    # depth_file = "/home/niu/下载/indoor_train-004/train/HR/04. Church/depth_vi/in_01_160225_160701_depth_vi.png"
    # depth_file = "/home/niu/下载/depth_images/800.png"
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_53.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/raw_png/in_k_00_160120_000001_rd.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/up_png/in_k_00_160120_000002_ud.png"
    depth_file = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_19336.png"  # depth_19336.png
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_11.png"
    # depth_file = "/home/niu/下载/02_cafe_2/2/17.01.19/1/raw_png/in_k_01_170119_000001_rd.png"

    a = time.time()
    for i in range(1):
        resized_depth, collision_map = process_depth_pipeline(depth_file)
    b = time.time() - a
    print(f"处理时间：{b:.3f}秒")
