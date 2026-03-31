import math
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from agent_encoder.utils import * # 移除了本地依赖，假设utils包含标准函数
from matplotlib import font_manager

font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为实际路径
try:
    font_prop = font_manager.FontProperties(fname=font_path)
    # 设置字体
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except FileNotFoundError:
    print(f"警告：未找到中文字体 '{font_path}'。标题可能显示为乱码。")


# 假设的常量 (基于代码上下文)
MIN_DEPTH = 5
MAX_DEPTH = 255
DRONE_HALF_SIZE_METERS = 0.2
fx = 240  # 假设值
fy = 240  # 假设值
MAX_DILATION_SIZE = 30


# 假设的缺失函数 (基于代码上下文)
def uint8_normalize(img, max_val=255):
    # 简单的归一化实现
    img_copy = img.astype(np.float32)  # 转换为浮点数进行计算
    if img_copy.max() > 0:
        # 改进归一化以处理 min/max 相同的情况
        min_val = img_copy.min()
        max_val_img = img_copy.max()
        if max_val_img == min_val:
            img_copy = np.zeros_like(img_copy)
        else:
            img_copy = (img_copy - min_val) / (max_val_img - min_val) * max_val
    return img_copy.astype(np.uint8)


def apply_drone_offset(img):
    # 假设这是一个占位符，实际实现可能更复杂
    return img


# -------------------------------------------------
# --- 你提供的原代码（稍作清理）---
# -------------------------------------------------

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
    plt.savefig("preprocess_results.png")  # 使用 savefig 替代 show
    plt.close()


def filter_edge_and_min_depth(depth_img, edge_width=2):
    """
    标记无效像素：图像边缘（宽度edge_width）和深度值 < MIN_DEPTH 的像素
    """
    assert depth_img.dtype == np.uint8 and len(depth_img.shape) == 2, "输入必须是uint8单通道深度图"
    zero_mask = (depth_img < MIN_DEPTH).astype(bool)
    edge_mask = np.zeros_like(zero_mask, dtype=bool)
    invalid_mask = zero_mask | edge_mask
    return invalid_mask.astype(np.uint8)


def resize_and_fill_depth(depth_map, half_plot=False):
    """
    调整深度图大小并填充零值区域，小于等于MIN_DEPTH的像素值加入零值掩膜
    """
    depth_map_copy = depth_map.copy()
    uint8_depth = uint8_normalize(depth_map_copy)
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


# -------------------------------------------------
# --- 关键修改：generate_edges_from_depth ---
# -------------------------------------------------
def generate_edges_from_depth(depth_map, zero_mask, edge_threshold_low=30, edge_threshold_high=50):
    """
    修改此函数以返回所有关键的中间图像，用于生成六面板图
    """
    # (D_fill 输入 'depth_map' 即是 D_fill)

    # 对zero_mask进行膨胀操作，生成 M'_zero
    dilated_zero_mask = dilate_zero_mask(zero_mask, dilation_iterations=1)

    # 将膨胀后的零值掩膜应用于深度图，生成 D'_fill
    # 这就是你想要的 D_fill_prime (depth_map_filled)
    depth_map_filled = depth_map.copy()
    depth_map_filled[dilated_zero_mask == 1] = 255

    # --- (b) 检测路径-直方图均衡化 (D_hist) ---
    hist_eq_depth = cv2.equalizeHist(depth_map_filled)
    normalized_depth = cv2.normalize(hist_eq_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- (c) 检测路径-Canny边缘 (E) ---
    edges = cv2.Canny(normalized_depth, edge_threshold_low, edge_threshold_high)

    kernel_size = 3
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # --- (d) 数据路径-腐蚀平滑 (D_filtered) ---
    filtered_depth_map = cv2.erode(depth_map_filled, kernel, iterations=1)

    # --- (e) 路径合并 (D_edge_initial) ---
    edge_depth_initial = depth_map_filled.copy()
    edge_depth_initial[:, :] = 255
    edge_depth_initial[edges != 0] = filtered_depth_map[edges != 0]

    # --- (f) 最终边缘深度图 (D_edge) ---
    # 再次使用 *原始* zero_mask 过滤掉零值区域的边缘
    edge_depth_final = edge_depth_initial.copy()
    edge_depth_final[zero_mask == 1] = 255

    # --- 修改返回 ---
    # 返回所有中间步骤的图像
    return depth_map_filled, hist_eq_depth, edges, filtered_depth_map, edge_depth_initial, edge_depth_final


def calculate_dilation_size(depth_value):
    """根据深度值计算膨胀大小"""
    if depth_value <= 0:
        return 1
    depth_in_meters = (depth_value / MAX_DEPTH) * 10.0
    if depth_in_meters == 0:
        return 1
    dilation_radius_x = (DRONE_HALF_SIZE_METERS * fx) / depth_in_meters
    dilation_radius_y = (DRONE_HALF_SIZE_METERS * fy) / depth_in_meters
    dilation_radius = int(max(dilation_radius_x, dilation_radius_y))
    return max(min(dilation_radius, MAX_DILATION_SIZE), 1)


def pixel_wise_dilation_optimized(depth_map, max_dilation=MAX_DILATION_SIZE):
    """逐像素膨胀深度图（保持原状）"""
    dilated_depth_map = np.full_like(depth_map, 0)
    valid_pixels = np.where((depth_map != 0))
    if valid_pixels[0].size == 0:
        dilated_depth_map[:] = 0
        return dilated_depth_map
    depth_values = depth_map[valid_pixels]
    y_coords, x_coords = valid_pixels
    dilation_sizes = np.array([calculate_dilation_size(d) for d in depth_values], dtype=np.int32)
    dilation_sizes = np.clip(dilation_sizes, 1, max_dilation)
    sort_indices = np.argsort(depth_values)[::-1]
    depth_values_sorted = depth_values[sort_indices]
    y_coords_sorted = y_coords[sort_indices]
    x_coords_sorted = x_coords[sort_indices]
    dilation_sizes_sorted = dilation_sizes[sort_indices]
    for y, x, d_size, depth_val in zip(y_coords_sorted, x_coords_sorted, dilation_sizes_sorted, depth_values_sorted):
        min_y = max(0, y - d_size)
        max_y = min(depth_map.shape[0], y + d_size + 1)
        min_x = max(0, x - d_size)
        max_x = min(depth_map.shape[1], x + d_size + 1)
        region = dilated_depth_map[min_y:max_y, min_x:max_x]
        mask = (region == 0) | (region > depth_val)
        updated_region = np.where(mask, depth_val, region)
        dilated_depth_map[min_y:max_y, min_x:max_x] = updated_region
    dilated_depth_map = np.where(dilated_depth_map > MIN_DEPTH, dilated_depth_map,
                                 MIN_DEPTH * np.ones_like(depth_map))
    return dilated_depth_map.astype(np.uint8)


def edge_dilation_optimized(edge_depth, max_dilation=MAX_DILATION_SIZE):
    """边缘膨胀（保持原状）"""
    dilated_edge_depth = np.full_like(edge_depth, 255)
    edge_pixels = np.where((edge_depth != 255) & (edge_depth > 5))
    if edge_pixels[0].size == 0:
        return dilated_edge_depth  # 没有边缘，返回全白
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
    """碰撞图生成（保持原状）"""
    result = np.zeros_like(depth_map1)
    non_zero_in_both = (depth_map1 != 0) & (depth_map2 != 0)
    non_zero_in_resized = (depth_map1 != 0) & (depth_map2 == 0)
    non_zero_in_dilated_edges = (depth_map1 == 0) & (depth_map2 != 0)
    result[non_zero_in_resized] = depth_map1[non_zero_in_resized]
    result[non_zero_in_dilated_edges] = depth_map2[non_zero_in_dilated_edges]
    result[non_zero_in_both] = np.minimum(depth_map1[non_zero_in_both], depth_map2[non_zero_in_both])
    return result


def generate_max(depth_map1, depth_map2):
    result = np.maximum(depth_map1, depth_map2)
    return result


# -------------------------------------------------
# --- 新增绘图函数：plot_dual_path_results (修改为 3x3) ---
# -------------------------------------------------
def plot_dual_path_results(
        d_resized, zero_mask, d_fill_prime,
        d_fill, d_hist, edges,
        d_filtered, d_edge_initial, d_edge_final
):
    """
    生成一个3x3的九面板图，展示完整的处理流程。
    （已增大字体并修正索引）
    """
    plt.figure(figsize=(30, 20))  # 调整尺寸以适应 3x3

    # 定义统一的字体大小
    title_fontsize = 50
    title_y_pos = -0.13

    # (1) 原始深度图 (Resized)
    plt.subplot(3, 3, 1)
    plt.imshow(d_resized, cmap='jet', vmin=0, vmax=255)
    plt.title("(a)原始深度图($D_{resized}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (2) 零值掩码 (M_zero)
    plt.subplot(3, 3, 2)
    plt.imshow(zero_mask, cmap="gray")
    plt.title("(b)零值掩码($M_{zero}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (3) 修复后深度图 (D_fill) <-- 修正索引
    plt.subplot(3, 3, 3)
    plt.imshow(d_fill, cmap='jet', vmin=0, vmax=255)
    plt.title("(c)填充深度图($D_{fill}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (4) 填充深度图副本 (D'_fill) <-- 修正索引
    plt.subplot(3, 3, 4)
    plt.imshow(d_fill_prime, cmap='jet', vmin=0, vmax=255)
    plt.title("(d)填充深度图副本($D^{\prime}_{fill}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (5) 检测路径-直方图均衡化 (D_hist)
    plt.subplot(3, 3, 5)
    plt.imshow(d_hist, cmap='jet', vmin=0, vmax=255)
    plt.title("(e)检测路径:直方图均衡化($D_{hist}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (6) 检测路径-Canny边缘 (E)
    plt.subplot(3, 3, 6)
    plt.imshow(edges, cmap="gray")
    plt.title("(f)检测路径:Canny边缘($E$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (7) 数据路径-腐蚀平滑 (D_filtered)
    plt.subplot(3, 3, 7)
    plt.imshow(d_filtered, cmap='jet', vmin=0, vmax=255)
    plt.title("(g)数据路径:腐蚀平滑($D_{filtered}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (8) 路径合并 (D_edge_initial)
    plt.subplot(3, 3, 8)
    plt.imshow(d_edge_initial, cmap="gray_r")  # gray_r (白底) 更适合
    plt.title("(h)初始边缘图($D_{edge\_init}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    # (9) 最终边缘深度图 (D_edge)
    plt.subplot(3, 3, 9)
    plt.imshow(d_edge_final, cmap="gray_r")  # gray_r (白底) 更适合
    plt.title("(i)边缘深度图($D_{edge}$)", fontsize=title_fontsize, y=title_y_pos)
    plt.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("2.完整边缘检测过程.png")  # 保存为新文件名
    plt.close()  # 关闭绘图，释放内存

# -------------------------------------------------
# --- 关键修改：process_depth_pipeline ---
# -------------------------------------------------
def process_depth_pipeline(depth_file_path):
    """
    深度处理流水线封装函数
    修改：调用新的 3x3 绘图函数，并移除冗余绘图
    """
    # 读取深度图
    oridepth_map = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
    if oridepth_map is None:
        raise ValueError(f"无法加载深度图文件：{depth_file_path}")
    oridepth_map = oridepth_map.astype(np.float32)

    # 预处理阶段
    # uint8_depth_resized 是你要求的目标图像
    zero_mask, uint8_depth_resized, uint8_depth_filled = resize_and_fill_depth(oridepth_map, False)

    # (a) D_fill (用于绘图)
    filled_for_plot = uint8_depth_filled.copy()

    # --- 修改解包 ---
    # 边缘生成阶段 - 调用修改后的函数
    (d_fill_prime,         # <-- 新增解包
     hist_eq_depth,
     edges,
     filtered_depth_map,
     edge_depth_initial,
     edge_depth
     ) = generate_edges_from_depth(uint8_depth_filled, zero_mask)

    # (注意：这里的 'edge_depth' 是(f)步骤的最终结果)
    dilated_edges = edge_dilation_optimized(edge_depth)
    dilated_edges = apply_drone_offset(dilated_edges)

    # 清理变量覆盖，使逻辑更清晰
    uint8_depth_filled_offset = apply_drone_offset(uint8_depth_filled)
    uint8_depth_filled_dilated = pixel_wise_dilation_optimized(uint8_depth_filled_offset)

    # 后处理阶段
    collisions = generate_coll(uint8_depth_filled_dilated, dilated_edges)

    d1 = uint8_depth_resized.astype(np.float32)
    d2 = collisions.astype(np.float32)
    diff = np.abs(d1 - d2)
    diff[diff < 0] = 0
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 修改函数调用 ---
    # 生成 3x3 九面板图
    plot_dual_path_results(
        uint8_depth_resized,  # 1. Resized Original (你的要求)
        zero_mask,            # 2. Zero Mask
        d_fill_prime,         # 3. D_fill_prime (NEW)
        filled_for_plot,      # 4. Inpainted (a)
        hist_eq_depth,        # 5. Equalized (b)
        edges,                # 6. Canny (c)
        filtered_depth_map,   # 7. Eroded (d)
        edge_depth_initial,   # 8. Merged (e)
        edge_depth            # 9. Final Edge (f)
    )


    return uint8_depth_resized, collisions


# -------------------------------------------------
# --- 主执行入口 (保持原状) ---
# -------------------------------------------------
if __name__ == "__main__":
    # depth_file = "/home/niu/下载/indoor_train-004/train/HR/02. Cafe/depth_vi/in_00_160315_165831_depth_vi.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/warp_png/in_k_00_160120_000001_wd.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/up_png/in_k_00_160120_000001_ud.png"
    # depth_file = "/home/niu/下载/depth_images/800.png"
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_11.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/raw_png/in_k_00_160120_000001_rd.png"
    # depth_file = "/home/niu/下载/03_claseeroom_1/1/16.01.20/1/up_png/in_k_00_160120_000002_ud.png"
    # depth_file = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_19336.png"  # depth_19336.png
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/utils/vae/data_test/depths/depth_image_0.png"

    # depth_file = "/home/niu/下载/02_cafe_2/2/17.01.19/1/raw_png/in_k_01_170119_000001_rd.png"     # 一开始用的真实
    depth_file = "/home/niu/下载/01. Warehouse/17.02.01/1/raw_png/in_k_03_170201_000630_rd.png"
    # depth_file = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png"        # 仿真

    a = time.time()
    try:
        for i in range(1):
            resized_depth, collision_map = process_depth_pipeline(depth_file)
        b = time.time() - a
        print(f"处理时间：{b:.3f}秒")
        print("绘图已保存为 '2.完整边缘检测过程.png'")

    except ValueError as e:
        print(e)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{depth_file}'。")
        print("请确保 'depth_file' 路径设置正确，或者取消注释代码中的虚拟图像创建部分来进行测试。")
    except Exception as e:
        print(f"发生错误: {e}")
        print("请确保 'depth_file' 路径设置正确，并且所有依赖项（如OpenCV, Matplotlib）都已安装。")