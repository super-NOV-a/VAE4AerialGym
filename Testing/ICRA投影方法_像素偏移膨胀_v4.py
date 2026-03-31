import numpy as np
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

# 创建网格
def create_meshgrid(height, width, cx, cy, fx, fy):
    x = np.arange(0, height, dtype=np.float32)
    y = np.arange(0, width, dtype=np.float32)

    x, y = np.meshgrid(y, x)
    z = np.ones((height, width))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x, y, z], axis=0)

# 深度图像转点云
def depth_to_pointcloud(depth_img, meshgrid, scale=1.0, offset_dist=5.0):
    x = meshgrid[0] * depth_img * scale
    y = meshgrid[1] * depth_img * scale
    z = meshgrid[2] * depth_img * scale
    z_pcl = z.copy()
    z_pcl[z_pcl < 1.0] = MAX_DIST

    range_img = np.sqrt(x ** 2 + y ** 2 + z ** 2)+ 1e-6
    z_offset = (1 - offset_dist / range_img) * z
    point_cloud = np.stack([x, y, z_pcl], axis=0)
    return point_cloud, z_offset

# 创建立方体网格
def create_cube_mesh(edges, point_cloud, edge_length=0.2):
    cube_mesh_list = []
    for i in range(edges.shape[0]):
        x_edge = edges[i, 1]
        y_edge = edges[i, 0]
        point_origin = point_cloud[:, y_edge, x_edge]
        cube_mesh_list.append(tm.creation.box(extents=[edge_length, edge_length, edge_length]))
        cube_mesh_list[-1].apply_translation(point_origin)
    return cube_mesh_list

# 处理深度图像并生成膨胀边缘
def process_depth_image(depth_img, edges, cam_params, zero_mask=None):
    # depth_img = cv2.resize(depth_img, (96, 54), interpolation=cv2.INTER_LINEAR)
    depth_img /= 25.5
    depth_img[depth_img < 0.2] = 0
    depth_img[depth_img > 10] = 10

    # 创建网格
    meshgrid = create_meshgrid(depth_img.shape[0], depth_img.shape[1], cam_params.cx, cam_params.cy, cam_params.fx,
                               cam_params.fy)
    # 转换为三维点云
    point_cloud, offset_depth = depth_to_pointcloud(depth_img, meshgrid, scale=1.0, offset_dist=DRONE_HALF_SIZE_METERS)

    # # 绘制3D点云
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(point_cloud[0].flatten(), point_cloud[1].flatten(), point_cloud[2].flatten(), s=0.1)
    # ax.set_title('3D Point Cloud')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.show()

    if zero_mask is not None:
        # 过滤掉zero_mask中的边缘点
        valid_edges = []
        for edge in edges:
            y, x = edge
            if zero_mask[y, x] == 0:  # 如果该位置不在zero_mask中，则保留该边缘点
                valid_edges.append(edge)
        edges = np.array(valid_edges)

    # 创建立方体网格
    cube_mesh_list = create_cube_mesh(edges, point_cloud, edge_length=ROBOT_EDGE_LENGTH)
    cube_mesh_aggregated = tm.util.concatenate(cube_mesh_list)

    # 创建 wp.Mesh 对象
    points = wp.array(np.array(cube_mesh_aggregated.vertices), dtype=wp.vec3, device="cuda:0")
    faces = wp.array(np.array(cube_mesh_aggregated.faces.flatten()), dtype=wp.int32, device="cuda:0")
    wp_mesh = wp.Mesh(points, faces)

    # 创建像素数组并启动绘制核函数
    pixels = wp.zeros(270 * 480, dtype=wp.float32, device="cuda:0")
    wp.launch(
                kernel=draw,
                dim=270 * 480,
                inputs=[wp_mesh.id, wp.vec3(0, 0, 0), 480, 270, pixels, cam_params.cx, cam_params.cy,
                cam_params.fx, cam_params.fy],
    )

    # 获取投射图像并归一化
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
        edges = np.array(list(zip(edges[0], edges[1])))

        # 过滤零值区域的边缘
        for i in range(edges.shape[0]):
            edge = edges[i]
            if zero_mask[edge[0], edge[1]] == 1:
                edges[i] = (0, 0)  # 标记为无效点

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

    # 根据相似三角形原理计算图像平面中的半径（像素）
    dilation_radius_x = (0.8*DRONE_HALF_SIZE_METERS * fx) / depth_in_meters
    dilation_radius_y = (0.8*DRONE_HALF_SIZE_METERS * fy) / depth_in_meters

    # 取较大的膨胀半径并转换为整数
    dilation_radius = max(dilation_radius_x, dilation_radius_y)

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

# 主函数
if __name__ == "__main__":
    # 示例深度图像和边缘点
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/utils/vae/data_test/depths/depth_image_2.png"
    depth_file = "/home/niu/下载/02_cafe_2/2/17.01.19/1/raw_png/in_k_01_170119_000001_rd.png"
    # depth_file = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_53.png"
    depth_img = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # 深度图像归一化
    depth_img = uint8_normalize(depth_img)
    depth_img = cv2.resize(depth_img, (480, 270), interpolation=cv2.INTER_LINEAR)

    depth_img_copy = depth_img.copy()

    cam_params = CamParams(cx=240, cy=135, fx=252.91646, fy=252.91646)

    edge_detector = EdgeDetector(threshold1=30, threshold2=50)
    # 边缘图像
    edges, edge_image = edge_detector.process_image(depth_img.astype(np.uint8))

    # 处理深度图像并生成膨胀边缘
    raycast_img, offset_depth = process_depth_image(depth_img.astype(np.float32), edges, cam_params)
    dilated_depth = pixel_wise_dilation_optimized(offset_depth)
    # depth_img_copy = apply_drone_offset(dilated_depth, to255=False)

    # final_iamge1 = np.min([dilated_depth, raycast_img], axis=0)
    # # 计算基础最终图像
    base_final_image1 = np.minimum(dilated_depth, raycast_img)
    # 应用阈值规则
    mask = depth_img <= 5
    final_image1 = np.where(mask, depth_img/25.5, base_final_image1)
    final_image2 = np.minimum(offset_depth, raycast_img)

    # 可视化结果
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 3, 1)
    plt.imshow(depth_img, cmap="gray")
    plt.title("Original Depth Image")
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.imshow(depth_img, cmap="gray")
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

    plt.subplot(3, 3, 6)
    plt.imshow(final_image1, cmap="gray")
    plt.title("Dilation + RayCast Image")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.imshow(final_image2, cmap="gray")
    plt.title("Offset + RayCast Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()