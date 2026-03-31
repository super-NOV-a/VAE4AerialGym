import cv2
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from matplotlib import font_manager
from agent_encoder.Display_depth_target_colls_dataset import PureDepthCollisionDataset
from agent_encoder.vae_image_a_test import DepthVAEReconstructor
from Testing.not_used.icra_原始方法 import CamParams, EdgeDetector, process_depth_image

# 手动指定字体路径
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'  # 替换为实际路径
font_prop = font_manager.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Evaluator:
    def __init__(self, depths_folder, colls_folder, image_size=(480, 270),
                 latent_dims=[32, 64, 128, 256],
                 model_types=['Beta_VAE', 'DC_VAE'],
                 beta_2_colls=False,
                 batch_size=8):
        # 注意：image_size 应该是 (height, width) 格式
        self.image_size = (image_size[1], image_size[0]) if len(image_size) == 2 else image_size
        self.min_depth = 15
        self.max_depth = 255
        self.latent_dims = latent_dims
        self.model_types = model_types
        self.beta_2_colls = beta_2_colls
        self.batch_size = batch_size

        # 创建数据集
        self.dataset = PureDepthCollisionDataset(
            depths_folder=depths_folder,
            colls_folder=colls_folder,
            transform=None,
            return_file_name=False
        )

        # 数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        self.cam_params = CamParams(cx=240, cy=135, fx=252.91646, fy=252.91646)
        self.edge_detector = EdgeDetector(threshold1=30, threshold2=50)

        # 预加载所有模型
        self.models = self._preload_models()

    def _preload_models(self):
        """预加载所有模型到内存中"""
        models = {}
        print("正在预加载所有模型...")

        for model_type in self.model_types:
            models[model_type] = {}
            for ld in self.latent_dims:
                print(f"加载模型: {model_type}, 潜在维度: {ld}")
                reconstructor = self._create_reconstructor(model_type, ld)
                models[model_type][ld] = reconstructor

        print("所有模型加载完成！")
        return models

    def _create_reconstructor(self, model_type, latent_dim):
        """根据模型类型和潜在维度创建VAE重建器实例"""
        if model_type == 'Beta_VAE':
            model_path = (f"/home/niu/workspaces/VAE_ws/agent_encoder/weights/"
                          f"beta_vae_beta100.0_LD_{latent_dim}_epoch_30.pth")
        elif model_type == 'DC_VAE':
            model_path = (f"/home/niu/workspaces/VAE_ws/agent_encoder/weights/"
                          f"dc_vae_beta100.0_LD_{latent_dim}_epoch_30.pth")
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        return DepthVAEReconstructor(
            model_path=model_path,
            latent_dim=latent_dim,
            image_size=self.image_size
        )

    def preprocess_depth_batch(self, depth_batch):
        """批量预处理深度图 - 修复形状问题"""
        batch_size = depth_batch.shape[0]
        # 修正：使用 (height, width) 而不是 (width, height)
        processed_batch = np.zeros((batch_size, self.image_size[0], self.image_size[1]), dtype=np.float32)

        for i in range(batch_size):
            depth_np = depth_batch[i].numpy().squeeze().astype(np.float32)

            # 预处理
            if np.max(depth_np) > 255:
                depth_np = depth_np * 255 / 7000

            depth_np[depth_np > self.max_depth] = self.max_depth
            depth_np[depth_np < self.min_depth] = 0

            # 调整大小 - cv2.resize 使用 (width, height) 但返回 (height, width)
            resized = cv2.resize(depth_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            processed_batch[i] = resized

        return processed_batch

    def preprocess_collision_batch(self, coll_batch):
        """批量预处理碰撞图 - 修复形状问题"""
        batch_size = coll_batch.shape[0]
        # 修正：使用 (height, width) 而不是 (width, height)
        processed_batch = np.zeros((batch_size, self.image_size[0], self.image_size[1]), dtype=np.float32)

        for i in range(batch_size):
            coll_np = coll_batch[i].numpy().squeeze().astype(np.float32)
            # 调整大小 - cv2.resize 使用 (width, height) 但返回 (height, width)
            resized = cv2.resize(coll_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            processed_batch[i] = resized

        return processed_batch

    def calculate_rmse(self, img1, img2):
        """计算均方根误差"""
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        if np.isnan(img1).any() or np.isnan(img2).any():
            return np.nan

        return np.sqrt(np.mean((img1 - img2) ** 2))

    def calculate_ssim(self, img1, img2):
        """
        计算结构相似性指数 (推荐版)
        结合了方法2的健壮性和方法1的标准数据范围设定
        """
        # 1. 类型转换，确保精度
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # 2. NaN 检查
        if np.isnan(img1).any() or np.isnan(img2).any():
            return np.nan

        # 3. 动态范围设定
        # 如果你的数据明确是 0-255 范围（如深度图转换后的图像），建议固定为 255
        # 如果数据可能是 0-1 范围的 float，则保留动态计算或通过参数传入
        data_range = 255.0

        # 4. 窗口大小自适应 (防止小图崩溃)
        # 尝试使用 7 (skimage默认)，但不超过图像短边的尺寸
        win_size = min(7, min(img1.shape) - 1)
        # 确保是奇数且至少为 3
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)

        try:
            ssim_value = ssim(
                img1,
                img2,
                data_range=data_range,
                win_size=win_size
            )
        except ValueError as e:
            print(f"SSIM计算错误: {e}")
            ssim_value = np.nan

        return ssim_value

    def clean_image_batch(self, image_batch):
        """批量清理图像中的无效值"""
        cleaned_batch = np.clip(image_batch, 0, 255)
        cleaned_batch = np.nan_to_num(cleaned_batch, nan=0.0, posinf=255.0, neginf=0.0)
        return cleaned_batch

    def evaluate_vae_batch(self, depth_batch, model_type, latent_dim):
        """批量处理 - 修复形状问题"""
        # 预处理深度图批次
        processed_depth_batch = self.preprocess_depth_batch(depth_batch)
        reconstructor = self.models[model_type][latent_dim]

        try:
            # 批量重建碰撞图
            reconstructed_batch = []
            for i in range(len(processed_depth_batch)):
                return_images = reconstructor.forward(processed_depth_batch[i])
                reconstructed_coll = (return_images[2] * 255).astype(np.float32)

                if self.beta_2_colls and model_type == "Beta_VAE":
                    # 边缘图像
                    edges, edge_image = self.edge_detector.process_image(reconstructed_coll.astype(np.uint8))
                    # 处理深度图像并生成膨胀边缘
                    raycast_img, offset_depth = process_depth_image(reconstructed_coll.astype(np.float32), edges,
                                                                    self.cam_params)
                    reconstructed_coll = 25.5 * np.minimum(offset_depth, raycast_img)

                reconstructed_batch.append(reconstructed_coll)

            reconstructed_batch = np.stack(reconstructed_batch)
            return self.clean_image_batch(reconstructed_batch)

        except Exception as e:
            print(f"{model_type} VAE批量处理错误: {e}")
            batch_size = len(depth_batch)
            # 修正：使用正确的形状 (height, width)
            return np.zeros((batch_size, self.image_size[0], self.image_size[1]), dtype=np.float32)

    def run_evaluation_optimized(self, max_samples=None):
        """优化后的评估函数"""
        # 初始化结果字典
        results = {
            model_type: {ld: {'rmse': [], 'ssim': []} for ld in self.latent_dims}
            for model_type in self.model_types
        }

        total_samples = len(self.dataset)
        if max_samples and max_samples < total_samples:
            from torch.utils.data import Subset
            subset_indices = list(range(max_samples))
            subset_dataset = Subset(self.dataset, subset_indices)
            dataloader = DataLoader(
                subset_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
            total_samples = max_samples
        else:
            dataloader = self.dataloader

        # 预先处理所有真实碰撞图
        print("正在预处理真实碰撞图...")
        all_orig_colls = []
        for batch_idx, (depth_batch, coll_batch) in enumerate(tqdm(dataloader, desc="预处理碰撞图")):
            orig_coll_batch = self.preprocess_collision_batch(coll_batch)
            # 确保不超出总样本限制
            for i in range(len(orig_coll_batch)):
                if len(all_orig_colls) < total_samples:
                    all_orig_colls.append(orig_coll_batch[i])

        # 对每个模型进行评估
        for model_type in self.model_types:
            for ld in self.latent_dims:
                print(f"\n正在评估模型: {model_type}, 潜在维度: {ld}")

                sample_idx = 0
                iterator = tqdm(enumerate(dataloader), total=len(dataloader),
                                desc=f"{model_type}-LD{ld}")

                for batch_idx, (depth_batch, _) in iterator:
                    # 批量重建碰撞图
                    recon_coll_batch = self.evaluate_vae_batch(depth_batch, model_type, ld)

                    # 对批次中的每个样本计算指标
                    batch_size = len(depth_batch)
                    for i in range(batch_size):
                        if sample_idx >= total_samples:
                            break

                        rmse_val = self.calculate_rmse(all_orig_colls[sample_idx], recon_coll_batch[i])
                        ssim_val = self.calculate_ssim(all_orig_colls[sample_idx], recon_coll_batch[i])

                        results[model_type][ld]['rmse'].append(rmse_val)
                        results[model_type][ld]['ssim'].append(ssim_val)
                        sample_idx += 1

                    # 更新进度条
                    if len(results[model_type][ld]['rmse']) > 0:
                        current_rmse = np.nanmean(results[model_type][ld]['rmse'][-batch_size:])
                        current_ssim = np.nanmean(results[model_type][ld]['ssim'][-batch_size:])

                        iterator.set_postfix({
                            'RMSE': f"{current_rmse:.2f}" if not np.isnan(current_rmse) else "NaN",
                            'SSIM': f"{current_ssim:.4f}" if not np.isnan(current_ssim) else "NaN"
                        })

        # 计算平均结果
        for model_type in self.model_types:
            for ld in self.latent_dims:
                valid_rmse = [x for x in results[model_type][ld]['rmse'] if not np.isnan(x)]
                valid_ssim = [x for x in results[model_type][ld]['ssim'] if not np.isnan(x)]

                results[model_type][ld]['avg_rmse'] = np.mean(valid_rmse) if valid_rmse else np.nan
                results[model_type][ld]['avg_ssim'] = np.mean(valid_ssim) if valid_ssim else np.nan

        return results

    # 保留其他方法...
    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "=" * 70)
        print("碰撞图重建性能评估 (多种VAE模型比较)")
        print("=" * 70)
        print(f"{'模型类型':<12} | {'潜在维度':<10} | {'Avg RMSE ↓':<12} | {'Avg SSIM ↑':<12} | {'Samples':<10}")
        print("-" * 70)

        for model_type in self.model_types:
            for ld in self.latent_dims:
                metrics = results[model_type][ld]
                rmse = metrics['avg_rmse']
                ssim_val = metrics['avg_ssim']
                samples = len(metrics['rmse'])

                rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "NaN"
                ssim_str = f"{ssim_val:.4f}" if not np.isnan(ssim_val) else "NaN"

                print(f"{model_type:<12} | {ld:<10} | {rmse_str:<12} | {ssim_str:<12} | {samples:<10}")

        print("=" * 70)

        with open("collision_evaluation_results.txt", "w") as f:
            f.write("碰撞图重建性能评估 (多种VAE模型比较)\n")
            f.write("=" * 70 + "\n")
            f.write(
                f"{'模型类型':<12} | {'潜在维度':<10} | {'Avg RMSE ↓':<12} | {'Avg SSIM ↑':<12} | {'Samples':<10}\n")
            f.write("-" * 70 + "\n")

            for model_type in self.model_types:
                for ld in self.latent_dims:
                    metrics = results[model_type][ld]
                    rmse = metrics['avg_rmse']
                    ssim_val = metrics['avg_ssim']
                    samples = len(metrics['rmse'])

                    rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "NaN"
                    ssim_str = f"{ssim_val:.4f}" if not np.isnan(ssim_val) else "NaN"

                    f.write(f"{model_type:<12} | {ld:<10} | {rmse_str:<12} | {ssim_str:<12} | {samples:<10}\n")

            f.write("=" * 70 + "\n")

        print("结果已保存到 collision_evaluation_results.txt")

    def evaluate_vae_single(self, depth_image, model_type, latent_dim):
        """单个图像评估，用于绘图等场景"""
        processed_depth = self.preprocess_depth_single(depth_image)
        reconstructor = self.models[model_type][latent_dim]

        try:
            return_images = reconstructor.forward(processed_depth)
            # return_images[2] 是VAE的重构输出
            # 对于 beta_vae，这是重构的深度图
            # 对于 dc_vae，这是生成的碰撞图
            raw_recon = (return_images[2] * 255).astype(np.float32)

            # 初始化 final_coll
            final_coll = raw_recon

            if self.beta_2_colls and model_type == "Beta_VAE":
                # 如果是 beta_vae，需要后处理生成碰撞图
                edges, edge_image = self.edge_detector.process_image(raw_recon.astype(np.uint8))
                raycast_img, offset_depth = process_depth_image(raw_recon.astype(np.float32), edges,
                                                                self.cam_params)
                final_coll = 25.5 * np.minimum(offset_depth, raycast_img)

            return self.clean_image_single(raw_recon), self.clean_image_single(final_coll)
        except Exception as e:
            print(f"{model_type} VAE处理错误: {e}")
            zeros = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)
            return zeros, zeros

    def preprocess_depth_single(self, image):
        """预处理单张深度图"""
        if np.max(image) > 255:
            image = image * 255 / 7000

        image[image > self.max_depth] = self.max_depth
        image[image < self.min_depth] = 0
        return cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

    def preprocess_collision_single(self, image):
        """预处理单张碰撞图"""
        return cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

    def clean_image_single(self, image):
        """清理单张图像中的无效值"""
        if np.isnan(image).any():
            image = np.nan_to_num(image, nan=0.0)

        if np.isinf(image).any():
            max_val = np.max(image[np.isfinite(image)])
            if np.isnan(max_val) or np.isinf(max_val):
                max_val = 255.0
            image = np.nan_to_num(image, posinf=max_val, neginf=0.0)

        image = np.clip(image, 0, 255)
        return image

    def plot_comparison(self, image_index=0, save_path="第三章对比2.png"):
        """
        绘制不同模型类型和潜在维度下的碰撞图重建比较
        修改1：左侧标签增加换行符
        修改2：字体保持 30 号
        """
        def_font_size = 40
        depth, coll = self.dataset[image_index]
        depth = depth.squeeze().astype(np.float32)
        coll = coll.squeeze().astype(np.float32)

        orig_depth = self.preprocess_depth_single(depth)
        orig_coll = self.preprocess_collision_single(coll)

        # === 修改部分开始：添加换行符 ===
        rows_config = []
        for model_type in self.model_types:
            if model_type == 'Beta_VAE' and self.beta_2_colls:
                # 对于 Beta-VAE，将标签分为两行：第一行是模型名，第二行是类型
                # 注意：r'' 字符串不支持 \n 转义，所以用 + 连接
                rows_config.append({
                    'model': model_type,
                    'type': 'recon',
                    'label': r'$\beta$-VAE' + '\n(Recon)'  # 这里加了换行
                })
                rows_config.append({
                    'model': model_type,
                    'type': 'coll',
                    'label': r'$\beta$-VAE' + '\n(Coll)'  # 这里加了换行
                })
            else:
                # 对于 DC-VAE
                rows_config.append({
                    'model': model_type,
                    'type': 'coll',
                    'label': 'DC-VAE\n(Coll)'  # 这里加了换行
                })
        # === 修改部分结束 ===

        n_rows = len(rows_config)
        n_cols = len(self.latent_dims) + 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols-0.5, 3 * n_rows))
        if n_rows == 1: axes = np.array([axes])

        col_titles = ["输入深度图", "重构目标图"] + [f"LD={ld}" for ld in self.latent_dims]

        # 设置顶部标题 (30号字体)
        for ax, col_title in zip(axes[0], col_titles):
            ax.set_title(col_title, fontsize=def_font_size, fontweight='bold', pad=10)

        # 遍历每一行进行绘制
        for i, config in enumerate(rows_config):
            ax_row = axes[i]
            model_type = config['model']
            row_type = config['type']

            # 设置左侧行标签 (30号字体 + 换行)
            ax_row[0].set_ylabel(
                config['label'],
                fontsize=def_font_size,
                fontweight='bold',
                rotation=0,
                labelpad=70,  # 保持距离
                multialignment='center',  # 多行文本内部居中对齐
                verticalalignment='center'  # 【新增】整个文本块相对于Y轴中心对齐
            )

            # 绘制基准列
            ax_row[0].imshow(orig_depth, cmap='jet')
            if row_type == 'recon':
                # ax_row[1].axis('off')
                ax_row[1].imshow(orig_depth, cmap='jet')
            else:
                ax_row[1].imshow(orig_coll, cmap='jet')

            # 绘制模型结果列
            for col_idx, ld in enumerate(self.latent_dims, start=2):
                raw_recon, final_coll = self.evaluate_vae_single(depth, model_type, ld)
                ax = ax_row[col_idx]

                if row_type == 'recon':
                    ax.imshow(raw_recon, cmap='jet')
                else:
                    ax.imshow(final_coll, cmap='jet')

        # 清理所有坐标轴
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                if ax.axison:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close(fig)
        print(f"碰撞图比较图已保存至: {save_path}")

if __name__ == "__main__":
    # 配置参数
    DEPTHS_FOLDER = "/home/niu/workspaces/VAE_ws/data_test/depths"
    COLLS_FOLDER = "/home/niu/workspaces/VAE_ws/data_test/colls"  # _target
    LATENT_DIMS = [32, 64, 128, 256]
    MODEL_TYPES = ['Beta_VAE', 'DC_VAE']

    # 初始化评估器
    evaluator = Evaluator(
        depths_folder=DEPTHS_FOLDER,
        colls_folder=COLLS_FOLDER,
        latent_dims=LATENT_DIMS,
        model_types=MODEL_TYPES,
        beta_2_colls=True,
        batch_size=128  # 根据内存调整
    )

    # 运行优化后的评估
    # results = evaluator.run_evaluation_optimized(max_samples=1000)
    # evaluator.print_results(results)

    # 绘制比较图
    evaluator.plot_comparison(image_index=105, save_path="2.dc-beta测试集对比.png")