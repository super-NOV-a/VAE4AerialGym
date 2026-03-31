import math
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from agent_encoder.utils import *
from matplotlib import font_manager
# 手动指定字体路径
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为实际路径
font_prop = font_manager.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def _plot_preprocess_results(original, mask, resized):
    """可视化预处理结果"""
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(uint8_normalize(original), cmap="gray", vmin=0, vmax=255)
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


def filter_edge_and_min_depth(depth_img, edge_width=2):
    """
    标记无效像素：图像边缘（宽度edge_width）和深度值 < MIN_DEPTH 的像素

    参数:
        depth_img (np.ndarray): uint8深度图，形状 (H, W)，范围 [0, 255]
        MIN_DEPTH (int): 深度阈值，小于此值的像素视为无效
        edge_width (int): 图像边缘的无效像素宽度（默认2）

    返回:
        invalid_mask (np.ndarray): 无效像素掩码（True=无效）
    """
    assert depth_img.dtype == np.uint8 and len(depth_img.shape) == 2, "输入必须是uint8单通道深度图"

    # --- 1. 标记深度过小的像素 ---
    zero_mask = (depth_img < MIN_DEPTH).astype(bool)

    # --- 2. 标记图像边缘像素 ---
    edge_mask = np.zeros_like(zero_mask, dtype=bool)

    # # 标记上下左右边缘（宽度为edge_width）
    # edge_mask[:edge_width, :] = True  # 上边缘
    # edge_mask[-edge_width:, :] = True  # 下边缘
    # edge_mask[:, :edge_width] = True  # 左边缘
    # edge_mask[:, -edge_width:] = True  # 右边缘

    # --- 合并掩码 ---
    invalid_mask = zero_mask | edge_mask

    return invalid_mask.astype(np.uint8)


def resize_and_fill_depth(depth_map, half_plot=False):
    """
    调整深度图大小并填充零值区域，小于等于MIN_DEPTH的像素值加入零值掩膜
    """
    depth_map_copy = depth_map.copy()   # 避免直接修改原始深度图
    uint8_depth = uint8_normalize(depth_map_copy)  # 范围0-255

    # 将小于等于0 或者像素数量少的 的像素值加入零值掩膜
    zero_mask = filter_edge_and_min_depth(uint8_depth)

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
#     return depth_map, edges

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
        dilated_depth_map (numpy.ndarray): 膨胀后的深度图，范围 [0, 255]
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

def generate_max(depth_map1, depth_map2):
    # 保存两图像中的最大值
    result = np.maximum(depth_map1, depth_map2)
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

    # 预处理阶段
    zero_mask, uint8_depth_resized, uint8_depth_filled = resize_and_fill_depth(oridepth_map, False)
    filled = uint8_depth_filled.copy()
    # 边缘生成阶段  边缘与填充图分别偏移
    _, edge_depth = generate_edges_from_depth(uint8_depth_filled, zero_mask)
    dilated_edges = edge_dilation_optimized(edge_depth)
    dilated_edges = apply_drone_offset(dilated_edges)   # 对膨胀的边缘图做一个偏移
    uint8_depth_filled = apply_drone_offset(uint8_depth_filled) # 对填充图做一个偏移
    uint8_depth_filled = pixel_wise_dilation_optimized(uint8_depth_filled)   # 对填充图做一个逐点膨胀


    # re_image = generate_max(uint8_depth_resized, uint8_depth_filled) # 原始图和填充图的最小值
    # 后处理阶段
    collisions = generate_coll(uint8_depth_filled, dilated_edges)
    # collisions = cv2.dilate(collisions, np.ones((3, 3), np.uint8), iterations=1)

    d1 = uint8_depth_resized.astype(np.float32)
    d2 = collisions.astype(np.float32)

    # 计算绝对差异（处理负值）
    diff = np.abs(d1 - d2)
    diff[diff < 0] = 0
    # 归一化处理
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # 可视化所有结果
    plot_fill_results(uint8_depth_resized, zero_mask, filled)
    # plot_chinese_results(uint8_depth_resized, zero_mask, edge_depth,
    #              dilated_edges, collisions, diff_norm)

    # plt.figure(figsize=(5, 5))
    # plt.imshow(collisions, cmap='jet')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    return uint8_depth_resized, collisions
def plot_fill_results(depth_resized, zero_mask, filled):
    """
    绘制填充结果的中文标题图像 (1x3)。包括原始深度图、零值掩码和填充深度图。
    该函数将图像保存为 "2.无效与填充.png"
    """
    title_font = 18
    plt.figure(figsize=(16, 3.6))

    plt.subplot(1, 3, 1)
    plt.imshow(uint8_normalize(depth_resized),cmap="jet",  vmin=0, vmax=255)
    plt.title("原始深度图", fontsize=title_font)
    plt.axis("off")


    plt.subplot(1, 3, 2)
    plt.imshow(zero_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("零值掩码", fontsize=title_font)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(uint8_normalize(filled),cmap="jet", vmin=0, vmax=255)
    plt.title("填充深度图", fontsize=title_font)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("2.无效与填充.png")  # 保存为新文件名
    plt.close()  # 关闭绘图，释放内存

def plot_chinese_results(depth_resized, zero_mask, edge_depth,
                 dilated_edges, collisions, diff):
    """
    绘制处理结果的中文标题图像 (2x3)。包括原始深度图、零值掩码、边缘深度图、膨胀边缘图、碰撞深度图和差异图。
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 3, 1)
    plt.imshow(uint8_normalize(depth_resized),cmap="plasma", vmin=0, vmax=255)
    plt.title("原始深度图")
    plt.axis("off")


    plt.subplot(2, 3, 2)
    plt.imshow(zero_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("零值掩码")
    plt.axis("off")


    plt.subplot(2, 3, 3)
    plt.imshow(edge_depth, cmap="gray_r")
    plt.title("边缘深度图")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(dilated_edges, cmap="plasma",vmin=0, vmax=255)
    plt.title("膨胀边缘图")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(collisions/255.0,cmap="plasma", vmin=0, vmax=1)
    plt.title("碰撞深度图")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(diff, cmap="plasma",vmin=0, vmax=255)
    plt.title("碰撞深度图与原始深度图的差异")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def plot_results(depth_resized, zero_mask, edge_depth,
                 dilated_edges, collisions, diff):
    """
    绘制处理结果的图像 (2x3)。包括原始深度图、零值掩码、边缘深度图、膨胀边缘图、碰撞深度图和差异图。
    """
    plt.figure(figsize=(25, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(uint8_normalize(depth_resized), cmap="jet", vmin=0, vmax=255)
    plt.title("Resized Original Depth")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(zero_mask, cmap="gray", vmin=0, vmax=1)
    plt.title("Zero Mask")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(edge_depth, cmap="gray_r")
    plt.title("Edge Depth")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(dilated_edges, cmap="jet", vmin=0, vmax=255)
    plt.title("Dilated Edges")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(collisions, cmap="jet", vmin=0, vmax=1)
    plt.title("Collisions")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(diff, cmap="jet", vmin=0, vmax=255)
    plt.title("Difference")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_resized_and_edge(depth_file_path):
    """
    一个简化的处理函数，仅用于获取缩放后的深度图和边缘深度图。

    参数:
        depth_file_path: 深度图文件路径
    返回:
        uint8_depth_resized: 缩放后的原始深度图 (uint8)
        edge_depth: 边缘深度图
    """
    # 读取深度图
    oridepth_map = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
    if oridepth_map is None:
        raise ValueError(f"无法加载深度图文件：{depth_file_path}")
    oridepth_map = oridepth_map.astype(np.float32)

    # 预处理阶段：获取缩放图和掩码
    zero_mask, uint8_depth_resized, uint8_depth_filled = resize_and_fill_depth(oridepth_map, False)

    # 边缘生成阶段
    _, edge_depth = generate_edges_from_depth(uint8_depth_filled, zero_mask)

    return uint8_depth_resized, edge_depth


def generate_edges_from_depth(depth_map, zero_mask, edge_threshold_low=30, edge_threshold_high=50,
                              return_intermediates=False):
    """
    修改：增加 return_intermediates 参数，返回 dilated_zero_mask (安全缓冲区)
    """
    # 1. 安全缓冲区 (Dilated Zero Mask)
    # 对 zero_mask 进行膨胀操作，扩大零值区域，形成保护边界
    dilated_zero_mask = dilate_zero_mask(zero_mask, dilation_iterations=1)

    # 2. D'_fill (填充深度图副本)
    # 将膨胀后的零值掩膜应用于深度图
    depth_map_filled = depth_map.copy()
    depth_map_filled[dilated_zero_mask == 1] = 255

    # 3. E (二值边缘图)
    hist_eq_depth = cv2.equalizeHist(depth_map_filled)
    normalized_depth = cv2.normalize(hist_eq_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(normalized_depth, edge_threshold_low, edge_threshold_high)

    # 4. D_filtered (平滑深度图)
    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filtered_depth_map = cv2.erode(depth_map_filled, kernel, iterations=1)

    # 5. D_edge_initial (初始边缘图)
    edge_depth = depth_map_filled.copy()
    edge_depth[:, :] = 255
    edge_depth[edges != 0] = filtered_depth_map[edges != 0]

    edge_depth_initial = edge_depth.copy()

    # 最终过滤
    edge_depth[zero_mask == 1] = 255

    if return_intermediates:
        # 返回列表新增 dilated_zero_mask
        return depth_map, edge_depth, depth_map_filled, edges, filtered_depth_map, edge_depth_initial, dilated_zero_mask

    return depth_map, edge_depth


def save_each_results(d_fill, d_fill_copy, e_edges, d_filtered, d_edge_initial, dilated_zero_mask):
    """
    修改：增加 dilated_zero_mask 参数并保存为 '安全缓冲区.png'
    """
    save_dir = "save_pics"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"创建文件夹: {save_dir}")

    def save_img(data, filename, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(6, 3.375))  # 16:9
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        print(f"已保存: {save_path}")

    # 1. D_fill
    save_img(uint8_normalize(d_fill), "填充深度图.png", 'jet', 0, 255)

    # [新增] 安全缓冲区 (Dilated Zero Mask) - Gray (二值图)
    save_img(dilated_zero_mask, "安全缓冲区.png", 'gray', 0, 1)

    # 2. D'_fill
    save_img(uint8_normalize(d_fill_copy), "填充深度图副本.png", 'jet', 0, 255)

    # 3. E
    save_img(e_edges, "二值边缘图.png", 'gray')

    # 4. D_filtered
    save_img(uint8_normalize(d_filtered), "平滑深度图.png", 'jet', 0, 255)

    # 5. D_edge_initial
    save_img(d_edge_initial, "初始边缘图.png", 'gray_r')

def plot_comparison(resized_1, edge_1, resized_2, edge_2):
    """
    绘制两张图像及其边缘深度图的对比图 (2x2)。

    参数:
        resized_1 (np.ndarray): 图像1的缩放深度图
        edge_1 (np.ndarray): 图像1的边缘深度图
        resized_2 (np.ndarray): 图像2的缩放深度图
        edge_2 (np.ndarray): 图像2的边缘深度图
    """
    plt.figure(figsize=(14, 9))  # 调整画布大小以便容纳2x2图像
    title_fontsize = 30
    # --- 图像 1 ---
    plt.subplot(2, 2, 1)
    plt.imshow(uint8_normalize(resized_1) ,cmap='jet', vmin=0, vmax=255)
    plt.title("(a) 图像1-原始深度图",fontsize=title_fontsize)
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(edge_1, cmap="gray_r")
    plt.title("(b) 图像1-边缘检测图", fontsize=title_fontsize)
    plt.axis("off")

    # --- 图像 2 ---
    plt.subplot(2, 2, 3)
    plt.imshow(uint8_normalize(resized_2), cmap="jet", vmin=0, vmax=255)
    plt.title("(c) 图像2-原始深度图",fontsize=title_fontsize)
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(edge_2, cmap="gray_r")
    plt.title("(d) 图像2-边缘检测图",fontsize=title_fontsize)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("2.边缘提取对比.png")  # 保存为新文件名
    plt.close()  # 关闭绘图，释放内存
# 该函数用于处理深度图像并生成碰撞图像  效果较好
# 1. 读取深度图像
# 2. 归一化深度图像（图1 深度图）
# 3. 生成填充图和零值掩码（图2 零值掩码），填充图偏置+填充图膨胀
# 4. 生成边缘图（图3边缘检测），边缘图偏置+边缘图膨胀（图4膨胀边缘）
# 5. 生成碰撞图像（图5 碰撞图=填充图膨胀+边缘图膨胀）
# 6. 碰撞图和原始归一化的深度图像的差异（图6 差异图）

if __name__ == "__main__":
    depth_file_1 = "/home/niu/下载/01. Warehouse/17.02.01/1/raw_png/in_k_03_170201_000630_rd.png"
    depth_file_2 = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png"

    print(f"--- 正在处理 图像 1 (并将保存中间过程) ---")
    oridepth_map_1 = cv2.imread(depth_file_1, cv2.IMREAD_UNCHANGED).astype(np.float32)
    zero_mask_1, uint8_depth_resized_1, uint8_depth_filled_1 = resize_and_fill_depth(oridepth_map_1, False)

    # 修改调用：接收 dilated_zero_mask
    _, edge_img_1, d_fill_copy, e_edges, d_filtered, d_edge_initial, dilated_zero_mask = generate_edges_from_depth(
        uint8_depth_filled_1, zero_mask_1, return_intermediates=True
    )

    print("正在保存图像1的中间处理结果...")
    save_each_results(
        d_fill=uint8_depth_filled_1,
        d_fill_copy=d_fill_copy,
        e_edges=e_edges,
        d_filtered=d_filtered,
        d_edge_initial=d_edge_initial,
        dilated_zero_mask=dilated_zero_mask  # 传入新增参数
    )

    print(f"\n--- 正在处理 图像 2 ---")
    oridepth_map_2 = cv2.imread(depth_file_2, cv2.IMREAD_UNCHANGED).astype(np.float32)
    zero_mask_2, uint8_depth_resized_2, uint8_depth_filled_2 = resize_and_fill_depth(oridepth_map_2, False)
    _, edge_img_2 = generate_edges_from_depth(uint8_depth_filled_2, zero_mask_2, return_intermediates=False)

    print("\n处理完成，正在绘制对比图...")
    plot_comparison(uint8_depth_resized_1, edge_img_1, uint8_depth_resized_2, edge_img_2)


# if __name__ == "__main__":
#     # --- 用于对比两张图像的新代码 ---
#
#     # 1. 定义两个图像文件路径
#     # 你可以修改这两个路径来对比你想要的任何两张图像
#     # depth_file = "/home/niu/下载/indoor_train-004/train/HR/02. Cafe/depth_vi/in_00_160315_165831_depth_vi.png"
#     depth_file = "/home/niu/下载/01. Warehouse/17.02.01/1/raw_png/in_k_03_170201_000630_rd.png"
#
#     depth_file_1 = "/home/niu/下载/01. Warehouse/17.02.01/1/raw_png/in_k_03_170201_000630_rd.png"
#     # depth_file_1 = "/home/niu/下载/indoor_train-004/train/HR/02. Cafe/depth_vi/in_00_160315_165831_depth_vi.png"    # 开始是用这个出的图
#     depth_file_2 = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png"
#
#     print(f"--- 正在处理 图像 1 ---")
#     print(f"路径: {depth_file_1}")
#     # 2. 处理第一张图像
#     start_time_1 = time.time()
#     resized_img_1, edge_img_1 = get_resized_and_edge(depth_file_1)
#     print(f"图像1 处理耗时: {time.time() - start_time_1:.3f} 秒")
#
#     print(f"\n--- 正在处理 图像 2 ---")
#     print(f"路径: {depth_file_2}")
#     # 3. 处理第二张图像
#     start_time_2 = time.time()
#     resized_img_2, edge_img_2 = get_resized_and_edge(depth_file_2)
#     print(f"图像2 处理耗时: {time.time() - start_time_2:.3f} 秒")
#
#     # 4. 绘制对比图
#     print("\n处理完成，正在绘制对比图...")
#     plot_comparison(resized_img_1, edge_img_1, resized_img_2, edge_img_2)
#
#     # --- 以下是原有的代码，已被注释掉 ---
#     a = time.time()
#     for i in range(1):
#         resized_depth, collision_map = process_depth_pipeline(depth_file)
#     b = time.time() - a
#     print(f"处理时间：{b:.3f}秒")
#
