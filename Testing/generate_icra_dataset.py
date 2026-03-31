import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import time
import sys
import argparse
import warp as wp
import warp.torch
import trimesh as tm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from agent_encoder.Display_pure_depth_dataset import DepthPureDataset, preprocess_image  # 从单独的文件导入数据集类
# show progress with tqdm
from tqdm import tqdm


wp.init()

MAX_DIST = 10.0
MIN_DIST = 0.2
ROBOT_EDGE_LENGTH = 0.5


# 该函数是绘制核函数，用于将网格投射到图像平面上
# 计算了网格的每个顶点在图像平面上的投射位置，并将其深度值存储在像素数组中

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

    # compute view ray
    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, 1.0))

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = 10.0    # color是深度值的初始值，默认为10.0

    if wp.mesh_query_ray(mesh, ro, rd, 50.0, t, u, v, sign, n, f):
        value = t * rd[2]
        if value < 0.2:
            color = -1.0
        else:
            color = t * rd[2]

    pixels[tid] = color


def create_meshgrid(height, width, cx, cy, fx, fy):
    """Creates a meshgrid.
    Parameters
    ----------
    height: int
        The height of the image.
    width: int
        The width of the image.
    cx: float
        The x-coordinate of the principal point.
    cy: float
        The y-coordinate of the principal point.
    fx: float
        The focal length in x-direction.
    fy: float
        The focal length in y-direction.
    Returns
    -------
    np.ndarray
        The meshgrid.
    """
    x = np.arange(0, height, dtype=np.float32)
    y = np.arange(0, width, dtype=np.float32)

    x, y = np.meshgrid(y, x)
    z = np.ones((height, width))
    x = (x - cx) / fx
    y = (y - cy) / fy
    return np.stack([x, y, z], axis=0)


def depth_to_pointcloud(depth_img, meshgrid, scale=1.0, offset_dist=5.0):
    """Converts a depth image to a point cloud.
    Parameters
    ----------
    depth_img: np.ndarray
        The depth image.
    meshgrid: np.ndarray
        The meshgrid of the depth image.
    scale: float
        The scale of the depth image.
    Returns
    -------
    np.ndarray
        The point cloud.
    """

    x = meshgrid[0] * depth_img * scale
    y = meshgrid[1] * depth_img * scale
    z = meshgrid[2] * depth_img * scale
    z_pcl = z.copy()
    z_pcl[z_pcl < 1.0] = MAX_DIST

    range_img = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    z_offset = (1 - offset_dist / range_img) * z
    point_cloud = np.stack([x, y, z_pcl], axis=0)
    return point_cloud, z_offset


def create_sphere_mesh(edges, point_cloud, radius=0.1):
    """Creates a sphere mesh at the centres of edges.
    Parameters
    ----------
    edges: np.ndarray
        The edge image.
    point_cloud: np.ndarray
        The point cloud.
    radius: float
        The radius of the sphere.
    Returns
    -------
    wp.Mesh
        The sphere mesh.
    """
    num_edges = edges.shape[0]
    sphere_mesh_list = []
    for i in range(num_edges):
        x_edge = edges[i, 1]
        y_edge = edges[i, 0]
        point_origin = point_cloud[:, y_edge, x_edge]

        if np.linalg.norm(point_origin) < 0.4:
            continue
        # print("point_origin: ", point_origin)
        sphere_mesh_list.append(tm.creation.icosphere(radius=radius, subdivisions=3))
        sphere_mesh_list[-1].apply_translation(point_origin)
    return sphere_mesh_list


def create_cube_mesh(edges, point_cloud, edge_length=0.2):
    """
    Creates a cube mesh at the centres of edges.
    Parameters
    ----------
    edges: np.ndarray
        The edge image.
    point_cloud: np.ndarray
        The point cloud.
    edge_length: float
        The edge length of the cube.
    Returns
    -------
    wp.Mesh
        The cube mesh.
    """
    num_edges = edges.shape[0]
    cube_mesh_list = []
    for i in range(num_edges):
        x_edge = edges[i, 1]
        y_edge = edges[i, 0]
        point_origin = point_cloud[:, y_edge, x_edge]
        cube_mesh_list.append(tm.creation.box(extents=[edge_length, edge_length, edge_length]))
        cube_mesh_list[-1].apply_translation(point_origin)
    return cube_mesh_list


class CamParams:
    def __init__(self, cx=240, cy=135, fx=252.91646, fy=252.91646):
        self.cx = cx
        self.cy = cy
        self.fx = fx
        self.fy = fy


class ImageProcessor:
    def __init__(self, min_depth=0.20, max_depth=10.0, scaling_factor=0.10, pixel_value_min_depth=-1.0,
                 pixel_value_max_depth=10.0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.pixel_value_min_depth = pixel_value_min_depth
        self.pixel_value_max_depth = pixel_value_max_depth
        self.scaling_factor = scaling_factor

    def process_image(self, image):
        image[image < self.min_depth] = self.pixel_value_min_depth
        image[image > self.max_depth] = self.pixel_value_max_depth
        image = image * self.scaling_factor
        image[image < 0.0] = 0.0
        image[image > 1.0] = 1.0
        return image


class EdgeDetector:
    def __init__(self, threshold1=30, threshold2=50):
        self.threshold1 = threshold1
        self.threshold2 = threshold2

    def process_image(self, image):
        edge_image = cv2.Canny(image, self.threshold1, self.threshold2)
        edges = np.where(edge_image > 0)
        # zip edges
        edges = np.array(list(zip(edges[0], edges[1])))
        for i in range(edges.shape[0]):
            edge = edges[i]
            # check min between neighbors
            neighbor_list = [
                (max(edge[0] - 1, 0), edge[1]),
                (min(edge[0] + 1, 269), edge[1]),
                (edge[0], max(0, edge[1] - 1)),
                (edge[0], min(479, edge[1] + 1)),
                (max(edge[0] - 2, 0), edge[1]),
                (min(edge[0] + 2, 269), edge[1]),
                (edge[0], max(0, edge[1] - 2)),
                (edge[0], min(479, edge[1] + 2)),
            ]
            min_depth = image[edge[0], edge[1]]
            if min_depth <= 0.0:
                for j in neighbor_list:
                    if image[j[0], j[1]] > 0.0:
                        min_depth = image[j[0], j[1]]
                        edge = j
                        break
            min_neighbor = (0, 0)
            for j in neighbor_list:
                if image[j[0], j[1]] < min_depth and image[j[0], j[1]] > 0.1:
                    min_depth = image[j[0], j[1]]
                    min_neighbor = j
            if min_depth < image[edge[0], edge[1]] and image[edge[0], edge[1]] > 0.1:
                edges[i] = min_neighbor
        return edges, edge_image


class CollisionImageProcessor:
    def __init__(self, cam_params, edge_detector, image_processor):
        self.cam_params = cam_params
        self.edge_detector = edge_detector
        self.image_processor = image_processor
        self.meshgrid = create_meshgrid(270, 480, cam_params.cx, cam_params.cy, cam_params.fx, cam_params.fy)

    def process_image(self, depth_img):
        if depth_img.shape != (270, 480):
            print("Error: Depth image shape is not (270, 480). The shape is: ", depth_img.shape)
            exit(1)

        # 原始深度图像
        depth_img_original = depth_img.copy()

        # 处理后的深度图像
        depth_img_processed = self.image_processor.process_image(depth_img.copy())

        # 偏移图像
        point_cloud, offset_image = depth_to_pointcloud(depth_img, self.meshgrid, scale=1.0, offset_dist=ROBOT_EDGE_LENGTH/2)

        # # 绘制3D点云
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(point_cloud[0].flatten(), point_cloud[1].flatten(), point_cloud[2].flatten(), s=0.1)
        # ax.set_title('3D Point Cloud')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # plt.show()

        # 边缘图像
        edges, edge_image = self.edge_detector.process_image((depth_img_processed * 255.0).astype(np.uint8))

        # 归一化偏移图像
        normalized_offset_image = self.image_processor.process_image(offset_image.copy())

        # # 边缘数量不足则返回 None
        # if len(edges) < 10:
        #     return None, None, None, None

        # 筛选边缘
        edges = edges[::5]

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
            inputs=[wp_mesh.id, wp.vec3(0, 0, 0), 480, 270, pixels, self.cam_params.cx, self.cam_params.cy,
                    self.cam_params.fx, self.cam_params.fy],
        )

        # 获取投射图像并归一化
        raycast_img = pixels.numpy().reshape(270, 480)
        normalized_raycast_image = self.image_processor.process_image(raycast_img.copy())

        # 结合碰撞图像
        combined_collision_image = np.minimum(normalized_offset_image, normalized_raycast_image)

        # 绘制所有图像
        plt.figure(figsize=(15, 10))

        # 原始深度图像
        plt.subplot(2, 4, 1)
        plt.imshow(depth_img_original, cmap='gray')
        plt.title('Original Depth Image')
        plt.axis('off')

        # 处理后的深度图像
        plt.subplot(2, 4, 2)
        plt.imshow(depth_img_processed, cmap='gray')
        plt.title('Processed Depth Image')
        plt.axis('off')

        # 偏移图像
        plt.subplot(2, 4, 3)
        plt.imshow(offset_image, cmap='gray')
        plt.title('Offset Image')
        plt.axis('off')

        # 边缘图像
        plt.subplot(2, 4, 4)
        plt.imshow(edge_image, cmap='gray')
        plt.title('Edge Image')
        plt.axis('off')

        # 归一化偏移图像
        plt.subplot(2, 4, 5)
        plt.imshow(normalized_offset_image, cmap='gray')
        plt.title('Normalized Offset Image')
        plt.axis('off')

        # 归一化投射图像
        plt.subplot(2, 4, 6)
        plt.imshow(normalized_raycast_image, cmap='gray')
        plt.title('Normalized Raycast Image')
        plt.axis('off')

        # 结合碰撞图像
        plt.subplot(2, 4, 7)
        plt.imshow(combined_collision_image, cmap='gray')
        plt.title('Combined Collision Image')
        plt.axis('off')

        # 调整子图布局
        plt.tight_layout()

        # 显示图像
        plt.show()

        # 返回结果
        return combined_collision_image, normalized_raycast_image, normalized_offset_image, edge_image


CX = 240
CY = 135
FX = 252.91646
FY = 252.91646

MAX_DEPTH = 10.0
MIN_DEPTH = 0.2

PICKLE_SAVE_FOLDER = "datasets/save_testing"


def main():
    cam_params = CamParams(CX, CY, FX, FY)
    image_processor = ImageProcessor(0.2, 10.0, 0.1, -1.0, 10.0)
    edge_detector = EdgeDetector(30, 50)
    collision_image_processor = CollisionImageProcessor(cam_params, edge_detector, image_processor)

    # print("Loading eval dataset from ", TFRECORD_TEST_FOLDER)

    depths_folder = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/utils/vae/data_test/depths"  # "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/utils/vae/data_test/colls"  # "/home/niu/workspaces/VAE_ws/datasets/colls_offset"

    # depths_folder = "/home/niu/workspaces/aerial_gym_ws/src/ori_aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images"
    # depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    # colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"

    # 准备数据集
    test_dataset = DepthPureDataset(depths_folder, transform=preprocess_image)

    # Define the data loaders
    test_loader = DataLoader(dataset=test_dataset, batch_size=1)
    print("Loaded data loaders")

    print("Number of training samples:", len(test_dataset))

    counter = 0
    for batch_idx, (depth_data, _ ) in tqdm(enumerate(test_loader), total=len(test_loader)):

        depth_data_torch = depth_data.to("cpu")
        # depth_data_torch /= MAX_DEPTH
        depth_data_torch[depth_data_torch < MIN_DEPTH / MAX_DEPTH] = 0.0
        depth_data_torch[depth_data_torch > 1.0] = 1.0

        depth_data_torch_true_depth = depth_data_torch.clone() * 10.0
        depth_data_torch_true_depth[depth_data_torch_true_depth < MIN_DEPTH] = -1.0

        # 返回值包括combined_collision_image, normalized_raycast_image, normalized_offset_image, edge_image
        processed_image, normalized_raycast_image, normalized_offset_image, edge_image = collision_image_processor.process_image(
            depth_data_torch_true_depth[0].detach().numpy().squeeze(0))
        if processed_image is None:
            continue

        processed_collision_image_full_depth = processed_image * MAX_DEPTH
        processed_collision_image_full_depth[processed_collision_image_full_depth < MIN_DEPTH] = 0.0
        depth_data_torch_true_depth[depth_data_torch_true_depth < MIN_DEPTH] = 0.0

        processed_collision_image_full_depth_uint16 = (255.0 * processed_collision_image_full_depth).astype(np.uint16)
        depth_data_torch_true_depth_uint16 = (255.0 * depth_data_torch_true_depth[0, 0].numpy()).astype(np.uint16)

        # # 绘制多个图像
        # plt.figure(figsize=(18, 12))
        #
        # # 原始深度图
        # plt.subplot(2, 3, 1)
        # original_depth = depth_data_torch_true_depth_uint16.squeeze()
        # plt.imshow(original_depth, cmap='gray')
        # plt.colorbar(label='Depth (0-255)')
        # plt.title('Original Depth Image')
        # plt.axis('on')
        #
        # # 处理后的碰撞图
        # plt.subplot(2, 3, 2)
        # processed_collision_depth = processed_collision_image_full_depth_uint16.squeeze()
        # plt.imshow(processed_collision_depth, cmap='gray')
        # plt.colorbar(label='Depth (0-255)')
        # plt.title('Processed Collision Image')
        # plt.axis('on')
        #
        # # 归一化的投射图像
        # plt.subplot(2, 3, 3)
        # normalized_raycast = (255.0 * normalized_raycast_image).astype(np.uint16)
        # plt.imshow(normalized_raycast, cmap='gray')
        # plt.colorbar(label='Depth (0-255)')
        # plt.title('Normalized Raycast Image')
        # plt.axis('on')
        #
        # # 归一化的偏移图像
        # plt.subplot(2, 3, 4)
        # normalized_offset = (255.0 * normalized_offset_image).astype(np.uint16)
        # plt.imshow(normalized_offset, cmap='gray')
        # plt.colorbar(label='Depth (0-255)')
        # plt.title('Normalized Offset Image')
        # plt.axis('on')
        #
        # # 边缘图像
        # plt.subplot(2, 3, 5)
        # edge_img = edge_image.astype(np.uint8)
        # plt.imshow(edge_img, cmap='gray')
        # plt.colorbar(label='Edge Intensity')
        # plt.title('Edge Image')
        # plt.axis('on')
        #
        # plt.tight_layout()
        # plt.show()  # 暂停以显示图像

        counter += 1
        # out_pickle_list.append([processed_collision_image_full_depth_uint16, depth_data_torch_true_depth_uint16])
        #
        # if len(out_pickle_list) >= 1000:
        #     pickle.dump(out_pickle_list,
        #                 open(PICKLE_SAVE_FOLDER + "collision_images_" + str(pickle_counter) + ".p", "wb"))
        #     print("Written out pickle file: ", pickle_counter)
        #     out_pickle_list = []
        #     pickle_counter += 1
    return


if __name__ == "__main__":
    main()
    print("Done.")
    sys.exit(0)