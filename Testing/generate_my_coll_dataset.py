import concurrent
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from agent_encoder.Display_pure_depth_dataset import DepthPureDataset
from torch.utils.data import DataLoader
import warp as wp
import trimesh as tm
from agent_encoder.utils import *

# 初始化Warp
wp.init()

# 全局常量
MAX_DIST = 10.0
ROBOT_EDGE_LENGTH = DRONE_SIZE_METERS

# 相机参数
class CamParams:
    def __init__(self, cx=240, cy=135, fx=252.91646, fy=252.91646):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy

# 绘制核函数
@wp.kernel
def draw(mesh: wp.uint64,
         cam_pos: wp.vec3,
         width: wp.int32,
         height: wp.int32,
         pixels: wp.array(dtype=wp.float32),
         cx: wp.float32,
         cy: wp.float32,
         fx: wp.float32,
         fy: wp.float32):
    tid = wp.tid()

    x = wp.float32(tid % width)
    y = wp.float32(tid // width)

    sx = float(x - cx) / fx
    sy = float(y - cy) / fy

    # 计算视图射线
    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, 1.0))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = 10.0  # 默认深度值

    if wp.mesh_query_ray(mesh, ro, rd, 50.0, t, u, v, sign, n, f):
        value = t * rd[2]
        if value < 0.2:
            color = -1.0
        else:
            color = t * rd[2]

    pixels[tid] = color


# 创建网格（向量化优化）
def create_meshgrid(height, width, cx, cy, fx, fy):
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing='ij'
    )
    z = torch.ones((height, width))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return torch.stack([x, y, z], dim=0)


# 深度图像转点云（PyTorch实现）
def depth_to_pointcloud(depth_img, meshgrid, scale=1.0, offset_dist=5.0):
    depth_tensor = torch.tensor(depth_img, dtype=torch.float32)
    x = meshgrid[0] * depth_tensor * scale
    y = meshgrid[1] * depth_tensor * scale
    z = meshgrid[2] * depth_tensor * scale

    z_pcl = z.clone()
    z_pcl[z_pcl < 1.0] = MAX_DIST

    range_img = torch.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-6
    z_offset = (1 - offset_dist / range_img) * z
    point_cloud = torch.stack([x, y, z_pcl], dim=0)
    return point_cloud.numpy(), z_offset.numpy()


# 创建立方体网格（优化）
def create_cube_mesh(edges, point_cloud, edge_length=0.2):
    if len(edges) == 0:
        return tm.Trimesh()

    # 正确提取点坐标 - 修复维度问题
    # point_cloud形状为(3, H, W), edges为(n,2) -> 提取后形状为(3, n)
    points = point_cloud[:, edges[:, 0], edges[:, 1]]

    # 转置为(n, 3)并转换为NumPy数组
    points = points.transpose(1, 0).numpy() if isinstance(points, torch.Tensor) else points.T

    # 批量创建立方体
    cube_size = [edge_length] * 3
    cubes = []

    for i in range(points.shape[0]):
        cube = tm.creation.box(extents=cube_size)
        cube.apply_translation(points[i])
        cubes.append(cube)

    return tm.util.concatenate(cubes)


# 处理深度图像（设备处理优化）
def process_depth_image(depth_img, edges, cam_params, zero_mask=None):
    depth_img = depth_img.copy()
    depth_img /= 25.5
    depth_img[depth_img < 0.2] = 0
    depth_img[depth_img > 10] = 10

    # 创建网格（使用PyTorch）
    meshgrid = create_meshgrid(depth_img.shape[0], depth_img.shape[1],
                               cam_params.cx, cam_params.cy,
                               cam_params.fx, cam_params.fy)

    # 转换为三维点云
    point_cloud, offset_depth = depth_to_pointcloud(depth_img, meshgrid,
                                                    scale=1.0,
                                                    offset_dist=DRONE_HALF_SIZE_METERS)

    # 过滤边缘点
    if zero_mask is not None and edges.size > 0:
        valid_mask = zero_mask[edges[:, 0], edges[:, 1]] == 0
        edges = edges[valid_mask]

    # 创建立方体网格
    if edges.size > 0:
        cube_mesh = create_cube_mesh(edges, point_cloud, edge_length=DRONE_SIZE_METERS)
        # 创建 wp.Mesh 对象
        points = wp.array(cube_mesh.vertices, dtype=wp.vec3, device="cuda:0")
        faces = wp.array(cube_mesh.faces.flatten(), dtype=wp.int32, device="cuda:0")
        wp_mesh = wp.Mesh(points, faces)
    else:
        # 空网格处理
        wp_mesh = wp.Mesh(wp.array([], dtype=wp.vec3, device="cuda:0"),
                          wp.array([], dtype=wp.int32, device="cuda:0"))

    # 创建像素数组并启动绘制核函数
    pixels = wp.zeros(270 * 480, dtype=wp.float32, device="cuda:0")
    wp.launch(
        kernel=draw,
        dim=270 * 480,
        inputs=[wp_mesh.id, wp.vec3(0, 0, 0), 480, 270, pixels,
                float(cam_params.cx), float(cam_params.cy),
                float(cam_params.fx), float(cam_params.fy)],
    )

    # 获取投射图像
    raycast_img = pixels.numpy().reshape(270, 480)
    return raycast_img, offset_depth

class EdgeDetector:
    def __init__(self, threshold1=30, threshold2=50):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def dilate_zero_mask(self, zero_mask, dilation_iterations=1):
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(zero_mask, kernel, iterations=dilation_iterations)
        return dilated_mask

    def process_image(self, image):
        # 生成零值掩模
        zero_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        zero_mask[image <= 5] = 1  # 标记零值区域

        # 对零值掩模进行膨胀操作，扩大零值区域
        dilated_zero_mask = self.dilate_zero_mask(zero_mask, dilation_iterations=1)

        # 将膨胀后的零值掩膜应用于深度图，填充零值区域边界
        depth_map_filled = image.copy()
        depth_map_filled[dilated_zero_mask == 1] = 255  # 或者使用周围像素的平均值进行填充

        # 应用直方图均衡化
        hist_eq_depth = cv2.equalizeHist(depth_map_filled.astype(np.uint8))  # 直方图均衡化

        # 归一化深度图
        depth_map_filled = cv2.normalize(hist_eq_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 应用Canny边缘检测
        edge_image = cv2.Canny(depth_map_filled, self.threshold1, self.threshold2)

        # 找到边缘点
        edges = np.where(edge_image > 0)
        if len(edges[0]) <= 5:
            return None, None

        edges = np.array(list(zip(edges[0], edges[1])))

        # 过滤零值区域的边缘
        for i in range(edges.shape[0]):
            edge = edges[i]
            if zero_mask[edge[0], edge[1]] == 1:
                edges[i] = (0, 0)  # 标记为无效点

        # 检查edges的维度
        if edges.ndim == 1:
            print(edges)

        # 移除无效点
        edges = edges[(edges[:, 0] != 0) | (edges[:, 1] != 0)]

        return edges, edge_image


def calculate_dilation_size(depth_in_meters):
    """根据深度值计算膨胀大小"""
    if depth_in_meters <= 0:
        return 1
    #
    # # 将深度图值转换为实际距离（米）
    # depth_in_meters = (depth_value / MAX_DEPTH) * 10.0

    # 根据相似三角形原理计算图像平面中的半径（像素）  fx = fy 故省略
    dilation_radius = (DRONE_HALF_SIZE_METERS * fx) / depth_in_meters
    # dilation_radius_y = (0.8*DRONE_HALF_SIZE_METERS * fy) / depth_in_meters
    # 取较大的膨胀半径并转换为整数
    # dilation_radius = max(dilation_radius_x, dilation_radius_y)

    # 限制在合理范围内
    return max(min(dilation_radius, MAX_DILATION_SIZE), 1)


def pixel_wise_dilation_optimized(depth_map):
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
    # dilation_sizes = np.clip(dilation_sizes, 1, max_dilation)

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
    dilated_depth_map = np.where(dilated_depth_map > 0.2, dilated_depth_map,
                                 0.2 * np.ones_like(depth_map))
    return dilated_depth_map    # .astype(np.uint8)


def save_images(final_image1, final_image2, save_path1, save_path2):
    # 保存为无压缩PNG（快速且通用）
    cv2.imwrite(save_path1, (final_image1 * 25.5).astype(np.uint8),    # [0, 10m] ->[0, 255]
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(save_path2, (final_image2 * 25.5).astype(np.uint8),
                [cv2.IMWRITE_PNG_COMPRESSION, 0])


def main():
    # print("Loading eval dataset from ", TFRECORD_TEST_FOLDER)

    # 预分配内存用于计时
    total_save_time = 0
    image_count = 0

    # depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    # colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"

    # 创建保存路径
    # output_dir1 = "/home/niu/workspaces/VAE_ws/datasets/colls_target"
    # output_dir2 = "/home/niu/workspaces/VAE_ws/datasets/colls_ICRA"

    depths_folder = "/home/niu/workspaces/VAE_ws/data_test/depths"
    # colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"

    # 创建保存路径
    output_dir1 = "/home/niu/workspaces/VAE_ws/data_test/colls"
    output_dir2 = "/home/niu/workspaces/VAE_ws/data_test/colls_ICRA"

    os.makedirs(output_dir1, exist_ok=True)
    os.makedirs(output_dir2, exist_ok=True)

    # 准备数据集
    test_dataset = DepthPureDataset(depths_folder, transform=False, return_file_name=True)

    # Define the data loaders
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    print("Loaded data loaders")

    print("Number of training samples:", len(test_dataset))

    cam_params = CamParams(cx=240, cy=135, fx=252.91646, fy=252.91646)

    edge_detector = EdgeDetector(threshold1=30, threshold2=50)
    show_results = False    # 测试时可以设为True以显示结果  todo

    counter = 0
    for batch_idx, (depth_data, _, file_name) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # 转换到CPU并转为NumPy
        depth_np = depth_data.squeeze(0).cpu().numpy().astype(np.float16)

        # 边缘检测（在CPU上）
        edges, edge_image = edge_detector.process_image(depth_np)

        if edges is None:
            final_image1 = depth_np
            final_image2 = depth_np

        else:
            # 处理深度图像（在CPU上处理）
            raycast_img, offset_depth = process_depth_image(
                depth_np, edges, cam_params
            )

            # 使用向量化膨胀
            dilated_depth = pixel_wise_dilation_optimized(offset_depth)

            # 计算最终图像
            # # 计算基础最终图像
            base_final_image1 = np.minimum(dilated_depth, raycast_img)
            # 应用阈值规则
            mask = depth_np <= 5
            final_image1 = np.where(mask, depth_np / 25.5, base_final_image1)
            final_image2 = np.minimum(offset_depth, raycast_img)

            if show_results:
                # 可视化结果
                plt.figure(figsize=(20, 12))
                plt.subplot(3, 3, 1)
                plt.imshow(depth_np, cmap="gray")
                plt.title("Original Depth Image")
                plt.axis("off")

                plt.subplot(3, 3, 2)
                plt.imshow(depth_np, cmap="gray")
                plt.scatter(edges[:, 1], edges[:, 0], color='red', s=1)
                plt.title("Edges on Depth Image")
                plt.axis("off")

                plt.subplot(3, 3, 3)
                plt.imshow(raycast_img, cmap="gray")
                plt.title("Expanded Edges Image")
                plt.axis("off")

                plt.subplot(3, 3, 4)
                plt.imshow(dilated_depth, cmap="gray")
                plt.title("Pixel-wise Dilation Depth Image")
                plt.axis("off")

                plt.subplot(3, 3, 5)
                plt.imshow(offset_depth, cmap="gray")
                plt.title("Depth Image with Drone Offset")
                plt.axis("off")

                plt.subplot(3, 3, 7)
                plt.imshow(final_image1, cmap="gray")
                plt.title("Dilation + RayCast Image")
                plt.axis("off")

                plt.subplot(3, 3, 8)
                plt.imshow(final_image2, cmap="gray")
                plt.title("Offset + RayCast Image")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

        # 生成保存路径
        save_path1 = os.path.join(output_dir1, file_name[0])
        save_path2 = os.path.join(output_dir2, file_name[0])

        # 并行保存图像并计时
        start_time = time.time()
        save_images(final_image1, final_image2, save_path1, save_path2)
        save_time = time.time() - start_time
        total_save_time += save_time
        counter += 1

    avg_save_time = total_save_time / image_count if image_count > 0 else 0
    print(f"\nProcessing complete. Total images: {image_count}")
    print(f"Average save time per image pair: {avg_save_time:.4f}s")
    print(f"Images saved to:\n  {output_dir1}\n  {output_dir2}")

    return


if __name__ == "__main__":
    main()
    print("Done.")
    sys.exit(0)