import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体

import numpy as np
import torch
import copy
import time
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# ===== 相机参数 =====
IMAGE_SIZE = (480, 270)  # (width, height)
CAMERA_HFOV_DEG = 87

ASPECT_RATIO = IMAGE_SIZE[0] / IMAGE_SIZE[1]
HFOV_RAD = np.radians(CAMERA_HFOV_DEG)
VFOV_RAD = 2 * np.arctan(np.tan(HFOV_RAD / 2) / ASPECT_RATIO)

fx = IMAGE_SIZE[0] / (2 * np.tan(HFOV_RAD / 2))
fy = IMAGE_SIZE[1] / (2 * np.tan(VFOV_RAD / 2))
cx, cy = IMAGE_SIZE[0] / 2, IMAGE_SIZE[1] / 2

print(f"相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


# ===== 1. 生成复杂深度图 (含任意形状) =====
def generate_complex_depth_map():
    H, W = IMAGE_SIZE[1], IMAGE_SIZE[0]
    y, x = torch.meshgrid(
        torch.linspace(0, H - 1, H, dtype=torch.float32),
        torch.linspace(0, W - 1, W, dtype=torch.float32),
        indexing='ij'
    )

    # 创建复杂几何形状
    sphere = torch.sqrt((x - W * 0.3) ** 2 + (y - H * 0.4) ** 2) / 50
    cube = torch.maximum(torch.abs(x - W * 0.7) / 40, torch.abs(y - H * 0.6) / 30)
    wave = 0.3 * torch.sin(x / 20 + y / 30)

    depth = 1 / (1 + 3 * ((x - cx) ** 2 + (y - cy) ** 2) / (W ** 2))
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 5 + 1

    # 添加复杂几何结构
    depth += 2 * torch.exp(-sphere ** 2)
    depth += 1.5 * torch.exp(-cube ** 2)
    depth += 0.5 * wave

    # 添加随机噪声
    depth += 0.1 * torch.randn(H, W)

    return depth.numpy()


depth_map = generate_complex_depth_map()


# ===== 2. 深度图转点云 =====
def depth_to_pointcloud(depth_map, fx, fy, cx, cy, stride=2):
    height, width = depth_map.shape
    points = []
    u_coords = []
    v_coords = []

    for v in range(0, height, stride):
        for u in range(0, width, stride):
            z = depth_map[v, u]
            if z > 0.1:
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                u_coords.append(u)
                v_coords.append(v)

    return np.array(points), np.array(u_coords), np.array(v_coords)


pointcloud, u_coords, v_coords = depth_to_pointcloud(depth_map, fx, fy, cx, cy, stride=2)
print(f"生成点云: {pointcloud.shape[0]}个点")


# ===== 3. 计算法向量 =====
def compute_normals(depth_map, fx, fy):
    height, width = depth_map.shape
    normals = np.zeros((height, width, 3))

    # 使用NumPy的梯度函数
    dz_du = np.gradient(depth_map, axis=1)
    dz_dv = np.gradient(depth_map, axis=0)

    for v in range(height):
        for u in range(width):
            dfdx = np.array([1.0, 0.0, dz_du[v, u]]) * depth_map[v, u] / fx
            dfdy = np.array([0.0, 1.0, dz_dv[v, u]]) * depth_map[v, u] / fy

            normal = np.cross(dfdx, dfdy)
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                normals[v, u] = normal / norm
            else:
                normals[v, u] = [0, 0, 1]  # 默认法向量

    return normals


normals_map = compute_normals(depth_map, fx, fy)


# ===== 4. 稳定各向异性高斯表示 =====
class StableAnisotropicGaussian:
    def __init__(self, position, covariance=None, normal=None, scale=None, opacity=0.8):
        self.position = np.array(position, dtype=np.float32)
        self.opacity = opacity

        # 修改初始化逻辑：基于点云分布构建初始形状
        if covariance is None:
            if scale is None:
                # 随机生成各向异性缩放因子
                scale_base = 0.03 + 0.04 * np.random.rand()
                scale = [
                    scale_base * (0.8 + 0.4 * np.random.rand()),  # X轴缩放
                    scale_base * (0.8 + 0.4 * np.random.rand()),  # Y轴缩放
                    scale_base * (0.05 + 0.1 * np.random.rand())  # Z轴缩放
                ]

            # 添加随机旋转扰动
            if normal is None:
                normal = np.array([0, 0, 1])
            normal /= np.linalg.norm(normal) + 1e-8
            self.z_axis = normal
            # 创建随机旋转矩阵
            angle = 2 * np.pi * np.random.rand()
            rot_axis = np.cross(normal, np.array([0, 1, 0]))
            if np.linalg.norm(rot_axis) < 1e-6:
                rot_axis = np.array([1, 0, 0])
            rot_axis /= np.linalg.norm(rot_axis)

            # 构建旋转矩阵 (Rodrigues公式)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            K = np.array([
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0]
            ])
            R = np.eye(3) + sin_a * K + (1 - cos_a) * K @ K

            # 构建协方差矩阵
            S = np.diag(np.square(scale))
            self.covariance = R @ S @ R.T
        else:
            self.covariance = covariance

    def get_scale_rotation(self):
        """从协方差矩阵分解出缩放和旋转（稳定版本）"""
        # 添加正则化确保正定
        regularized_cov = self.covariance + np.eye(3) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(regularized_cov)

        # 确保特征值为正
        scales = np.sqrt(np.maximum(eigvals, 1e-6))
        rotation = eigvecs

        # 确保右手坐标系
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1

        return scales, rotation

    def adjust_for_shape(self, points):
        """增强形状适应能力"""
        if len(points) < 5:
            return

        # 计算点云协方差
        cov = np.cov(points.T) + np.eye(3) * 1e-6

        # 分解特征值/特征向量
        eigvals, eigvecs = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sorted_idx]
        eigvecs = eigvecs[:, sorted_idx]

        # 计算形状复杂度因子
        shape_complexity = np.clip(eigvals[0] / (eigvals[2] + 1e-6), 1.0, 10.0)

        # 构建新的协方差矩阵
        new_cov = eigvecs @ np.diag(eigvals * (1.0 + 0.5 * shape_complexity)) @ eigvecs.T

        # 混合新旧协方差 (保留部分原始形状)
        self.covariance = 0.7 * new_cov + 0.3 * self.covariance

        # 确保法向量方向一致
        main_axis = eigvecs[:, 0]
        if np.dot(main_axis, self.z_axis) < 0:
            self.z_axis *= -1

    def optimize_parameters(self, gradient, lr=0.01, iteration=1):
        """增强形状优化能力"""
        decayed_lr = lr / (1 + 0.1 * iteration)

        # 位置更新
        self.position -= decayed_lr * gradient[:3]

        # 协方差更新 - 允许更大变化
        cov_grad = gradient[3:12].reshape(3, 3)

        # 放宽梯度限制 (从0.1->0.3)
        max_grad_norm = 0.3
        grad_norm = np.linalg.norm(cov_grad)
        if grad_norm > max_grad_norm:
            cov_grad = cov_grad * (max_grad_norm / grad_norm)

        self.covariance -= decayed_lr * cov_grad
        self.covariance = (self.covariance + self.covariance.T) / 2

        # 添加形状多样性正则化
        if iteration > 1:
            eigvals, _ = np.linalg.eigh(self.covariance)
            min_eigval = np.min(eigvals)
            if min_eigval < 1e-4:  # 防止过度扁平化
                self.covariance += np.eye(3) * (1e-4 - min_eigval)

        self.covariance = self.make_positive_definite(self.covariance)

    def make_positive_definite(self, matrix):
        """确保矩阵正定"""
        # 添加小单位矩阵正则化
        regularized = matrix + np.eye(3) * 1e-6

        # 特征值分解
        eigvals, eigvecs = np.linalg.eigh(regularized)

        # 截断负特征值
        eigvals = np.maximum(eigvals, 1e-6)

        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def project_to_screen(self, camera_params):
        """投影到屏幕空间并计算影响范围（高效版本）"""
        fx, fy, cx, cy = camera_params
        x, y, z = self.position

        # 投影坐标
        u = (x * fx / z) + cx
        v = (y * fy / z) + cy

        # 计算影响半径 (基于最大缩放)
        scales, _ = self.get_scale_rotation()
        max_scale = np.max(scales)

        # 限制半径范围
        radius = min(max(int(max_scale * fx / z * 2), 3), 30)

        return u, v, radius


# ===== 5. 基于几何复杂度的自适应高斯生成 =====
def generate_adaptive_gaussians(pointcloud, u_coords, v_coords, normals_map, target_count=100):
    """根据几何复杂度生成自适应高斯"""
    # 第一步：初始聚类
    kmeans = KMeans(n_clusters=target_count // 2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pointcloud)
    centers = kmeans.cluster_centers_

    gaussians = []
    for idx, center in enumerate(centers):
        # 收集当前聚类的点
        cluster_points = pointcloud[labels == idx]

        # 计算平均法向量
        cluster_normals = []
        for i in np.where(labels == idx)[0]:
            v, u = int(v_coords[i]), int(u_coords[i])
            cluster_normals.append(normals_map[v, u])

        mean_normal = np.mean(cluster_normals, axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-8

        # 修改：添加形状随机性因子
        shape_variation = 0.3 + 0.7 * np.random.rand()  # [0.3, 1.0]

        # 创建高斯 - 传入形状变化因子
        gaussian = StableAnisotropicGaussian(
            position=center,
            normal=mean_normal,
            scale=[0.05 * shape_variation,
                   0.05 * (1.2 - 0.4 * shape_variation),
                   0.01 * (0.5 + shape_variation)]
        )

        # 根据点云分布调整形状 (保留)
        gaussian.adjust_for_shape(cluster_points)

        # 根据点云分布调整形状
        gaussian.adjust_for_shape(cluster_points)
        gaussians.append(gaussian)

    # 第二步：在几何复杂区域增加高斯
    complexity_map = np.zeros(IMAGE_SIZE[::-1])

    # 计算几何复杂度（基于法向量变化）
    for v in range(1, IMAGE_SIZE[1] - 1):
        for u in range(1, IMAGE_SIZE[0] - 1):
            normal_center = normals_map[v, u]
            neighbors = [
                normals_map[v - 1, u], normals_map[v + 1, u],
                normals_map[v, u - 1], normals_map[v, u + 1]
            ]
            # 计算法向量之间的角度差异
            diff = 0
            for n in neighbors:
                # 点积并夹紧到[-1,1]避免浮点误差
                cos_theta = np.dot(normal_center, n)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                diff += np.arccos(cos_theta)
            complexity_map[v, u] = diff

    # 在复杂区域采样额外的高斯
    # 找到高复杂度区域（前10%）
    threshold = np.percentile(complexity_map, 90)
    high_complexity_mask = complexity_map > threshold

    # 收集高复杂度区域内的点
    high_complexity_points = []
    high_complexity_normals = []

    for i in range(len(pointcloud)):
        v_idx, u_idx = int(v_coords[i]), int(u_coords[i])
        # 确保坐标在图像范围内
        if 0 <= v_idx < IMAGE_SIZE[1] and 0 <= u_idx < IMAGE_SIZE[0]:
            if high_complexity_mask[v_idx, u_idx]:
                high_complexity_points.append(pointcloud[i])
                high_complexity_normals.append(normals_map[v_idx, u_idx])

    # 如果有高复杂度点，则添加额外高斯
    if high_complexity_points:
        extra_count = min(target_count // 2, len(high_complexity_points))
        extra_indices = np.random.choice(len(high_complexity_points), size=extra_count, replace=False)

        for idx in extra_indices:
            pt = high_complexity_points[idx]
            normal = high_complexity_normals[idx]
            gaussian = StableAnisotropicGaussian(position=pt, normal=normal)
            gaussians.append(gaussian)
        print(f"添加额外高斯: {extra_count}个")
    else:
        print("没有找到高复杂度区域")

    print(f"生成自适应高斯: {len(gaussians)}个 (基础: {len(centers)}, 额外: {len(gaussians) - len(centers)})")
    return gaussians


# 创建自适应高斯
gaussians = generate_adaptive_gaussians(pointcloud, u_coords, v_coords, normals_map, target_count=80)


# ===== 6. 高效深度渲染器 =====
def efficient_render_depth(gaussians, camera_params, img_size):
    fx, fy, cx, cy = camera_params
    h, w = img_size
    depth_img = np.zeros((h, w))
    weight_img = np.zeros((h, w))

    # 预计算索引网格
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

    for g in gaussians:
        # 投影到屏幕空间
        u_center, v_center, radius = g.project_to_screen(camera_params)

        # 计算影响区域
        u_min = max(0, int(u_center - radius))
        u_max = min(w, int(u_center + radius) + 1)
        v_min = max(0, int(v_center - radius))
        v_max = min(h, int(v_center + radius) + 1)

        # 只处理有效区域
        if u_min >= u_max or v_min >= v_max:
            continue

        # 创建局部网格
        local_u = u_grid[v_min:v_max, u_min:u_max]
        local_v = v_grid[v_min:v_max, u_min:u_max]

        # 计算局部坐标
        local_coords = np.stack([local_u, local_v], axis=-1)

        # 计算距离中心点的距离
        center = np.array([u_center, v_center])
        distances = np.linalg.norm(local_coords - center, axis=-1)

        # 创建距离掩码
        mask = distances <= radius

        # 3D位置反投影（批量计算）
        z_est = g.position[2]  # 使用高斯中心深度作为估计
        x = (local_u[mask] - cx) * z_est / fx
        y = (local_v[mask] - cy) * z_est / fy
        points = np.stack([x, y, np.full_like(x, z_est)], axis=-1)

        # 计算马氏距离（稳定版本）
        delta = points - g.position
        inv_cov = np.linalg.inv(g.covariance + np.eye(3) * 1e-6)

        # 批量计算马氏距离
        mahalanobis = np.sqrt(np.einsum('ni,ij,nj->n', delta, inv_cov, delta))
        mahalanobis = np.clip(mahalanobis, 0, 3)  # 限制在3σ内

        # 计算权重
        weights = g.opacity * np.exp(-0.5 * mahalanobis ** 2)

        # 更新深度和权重图
        valid_v = local_v[mask]
        valid_u = local_u[mask]

        depth_img[valid_v, valid_u] += weights * g.position[2]
        weight_img[valid_v, valid_u] += weights

    # 归一化处理
    valid_mask = weight_img > 1e-6
    depth_img[valid_mask] /= weight_img[valid_mask]
    depth_img[~valid_mask] = 0

    return depth_img


# 初始渲染
start_time = time.time()
rendered_depth = efficient_render_depth(
    gaussians,
    camera_params=(fx, fy, cx, cy),
    img_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
)
render_time = time.time() - start_time
print(f"深度渲染完成: {render_time:.2f}秒, {len(gaussians)}个高斯")


# ===== 7. 稳定参数优化循环 =====
def stable_optimize_gaussians(gaussians, target_depth, camera_params, img_size, num_iter=3):
    """稳定版本高斯参数优化"""
    optimized = copy.deepcopy(gaussians)
    h, w = img_size
    fx, fy, cx, cy = camera_params

    prev_loss = float('inf')  # 用于监控损失变化

    for iter_idx in range(num_iter):
        # 渲染当前深度
        current_depth = efficient_render_depth(optimized, camera_params, img_size)

        # 计算损失和梯度
        total_loss = 0
        valid_pixels = 0

        for g_idx, g in enumerate(optimized):
            # 获取高斯投影区域
            u_center, v_center, radius = g.project_to_screen(camera_params)
            u_min = max(0, int(u_center) - radius)
            u_max = min(w - 1, int(u_center) + radius)
            v_min = max(0, int(v_center) - radius)
            v_max = min(h - 1, int(v_center) + radius)

            # 跳过无效区域
            if u_min >= u_max or v_min >= v_max:
                continue

            grad_pos = np.zeros(3)
            grad_cov = np.zeros((3, 3))
            pixel_count = 0

            # 只计算有效区域
            for v in range(v_min, v_max + 1):
                for u in range(u_min, u_max + 1):
                    if current_depth[v, u] == 0 or target_depth[v, u] == 0:
                        continue

                    # 计算反投影3D点
                    z_est = current_depth[v, u]
                    x = (u - cx) * z_est / fx
                    y = (v - cy) * z_est / fy
                    point = np.array([x, y, z_est])

                    # 计算马氏距离（稳定版本）
                    delta = point - g.position
                    inv_cov = np.linalg.inv(g.covariance + np.eye(3) * 1e-6)

                    # 计算深度误差（带截断）
                    depth_error = z_est - target_depth[v, u]
                    depth_error = np.clip(depth_error, -1.0, 1.0)  # 防止大误差

                    # 位置梯度（带梯度裁剪）
                    pos_grad = 2 * depth_error * inv_cov @ delta
                    if np.linalg.norm(pos_grad) > 0.5:
                        pos_grad = pos_grad * 0.5 / np.linalg.norm(pos_grad)
                    grad_pos += pos_grad

                    # 协方差梯度（简化且稳定）
                    cov_grad = 0.05 * depth_error * np.outer(delta, delta) @ inv_cov
                    if np.linalg.norm(cov_grad) > 0.1:
                        cov_grad = cov_grad * 0.1 / np.linalg.norm(cov_grad)
                    grad_cov += cov_grad

                    pixel_count += 1
                    total_loss += depth_error ** 2
                    valid_pixels += 1

            if pixel_count > 0:
                # 归一化梯度
                grad_pos /= pixel_count
                grad_cov /= pixel_count

                # 组合梯度向量 [位置(3), 协方差(9)]
                gradient = np.concatenate([grad_pos, grad_cov.flatten()])

                # 更新参数
                g.optimize_parameters(gradient, lr=0.05, iteration=iter_idx)

        if valid_pixels > 0:
            avg_loss = total_loss / valid_pixels
            print(f"迭代 {iter_idx + 1}/{num_iter}, 损失: {avg_loss:.6f}")
            # 如果损失开始增加，提前停止
            if iter_idx > 0 and avg_loss > prev_loss * 1.5:
                print(f"损失增加，提前停止优化")
                break
            prev_loss = avg_loss
        else:
            print(f"迭代 {iter_idx + 1}/{num_iter}, 无有效像素")
            break

    return optimized


# 优化高斯参数
print("开始优化高斯参数...")
start_opt = time.time()
optimized_gaussians = stable_optimize_gaussians(
    gaussians,
    depth_map,
    camera_params=(fx, fy, cx, cy),
    img_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
)
opt_time = time.time() - start_opt
print(f"参数优化完成: {opt_time:.2f}秒")

# 优化后渲染
optimized_depth = efficient_render_depth(
    optimized_gaussians,
    camera_params=(fx, fy, cx, cy),
    img_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
)

# ===== 8. 结果可视化 =====
fig = plt.figure(figsize=(18, 12))

# 原始深度图
ax1 = fig.add_subplot(231)
ax1.imshow(depth_map, cmap='viridis')
ax1.set_title("原始深度图")

# 初始渲染结果
ax2 = fig.add_subplot(232)
ax2.imshow(rendered_depth, cmap='viridis')
ax2.set_title(f"初始渲染 ({len(gaussians)}个高斯)")

# 优化后渲染结果
ax3 = fig.add_subplot(233)
ax3.imshow(optimized_depth, cmap='viridis')
ax3.set_title(f"优化后渲染 ({len(optimized_gaussians)}个高斯)")

# 差异比较
ax4 = fig.add_subplot(234)
diff_initial = np.abs(depth_map - rendered_depth)
diff_initial[rendered_depth == 0] = 0
ax4.imshow(diff_initial, cmap='hot', vmin=0, vmax=1)
ax4.set_title("初始差异图")

ax5 = fig.add_subplot(235)
diff_optimized = np.abs(depth_map - optimized_depth)
diff_optimized[optimized_depth == 0] = 0
ax5.imshow(diff_optimized, cmap='hot', vmin=0, vmax=1)
ax5.set_title("优化后差异图")

# 高斯分布可视化
ax6 = fig.add_subplot(236, projection='3d')
positions = np.array([g.position for g in optimized_gaussians])
scales = np.array([g.get_scale_rotation()[0] for g in optimized_gaussians])

# 随机采样100个高斯可视化
if len(positions) > 0:
    sample_count = min(100, len(positions))
    sample_idx = np.random.choice(len(positions), sample_count, replace=False)

    for idx in sample_idx:
        g = optimized_gaussians[idx]
        pos = positions[idx]
        scale = scales[idx]

        # 创建椭球
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        x = scale[0] * np.outer(np.cos(u), np.sin(v))
        y = scale[1] * np.outer(np.sin(u), np.sin(v))
        z = scale[2] * np.outer(np.ones_like(u), np.cos(v))

        # 旋转和平移
        _, rotation = g.get_scale_rotation()
        for i in range(len(x)):
            for j in range(len(x[0])):
                point = np.array([x[i, j], y[i, j], z[i, j]])
                rotated = rotation @ point
                x[i, j], y[i, j], z[i, j] = rotated + pos

        ax6.plot_surface(x, y, z, alpha=0.1, color='blue')

ax6.set_title("3D高斯椭球分布")
ax6.set_xlabel("X")
ax6.set_ylabel("Y")
ax6.set_zlabel("Z")

plt.tight_layout()
plt.show()

# 性能报告
print("\n===== 性能报告 =====")
print(f"初始高斯数量: {len(gaussians)}")
print(f"点云压缩率: {len(gaussians) / len(pointcloud) * 100:.2f}%")
print(f"初始渲染时间: {render_time:.4f}秒")
print(f"优化后渲染时间: {time.time() - start_opt - render_time:.4f}秒")
print(f"初始MAE: {np.mean(diff_initial):.4f}")
print(f"优化后MAE: {np.mean(diff_optimized):.4f}")