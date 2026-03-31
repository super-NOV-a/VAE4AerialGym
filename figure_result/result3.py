import os
import cv2
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
from matplotlib import font_manager

# 假设项目结构中包含这些模块
from agent_encoder.Display_depth_target_colls_dataset import PureDepthCollisionDataset
from agent_encoder.vae_image_a_test import DepthVAEReconstructor
from Testing.not_used.icra_原始方法 import CamParams, EdgeDetector, process_depth_image
from agent_encoder.Net.AE import AE

# 手动指定字体路径 (如果需要绘图)
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'
if os.path.exists(font_path):
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


def set_seed(seed=42):
    """
    固定所有随机种子以保证实验可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子已固定为: {seed}")


class DepthAEReconstructor:
    """
    AE 模型重建器封装
    """

    def __init__(self, model_path, latent_dim, image_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.model = AE(input_dim=1, latent_dim=latent_dim).to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"警告: 找不到 AE 模型权重文件: {model_path}")

        self.model.eval()

    def forward(self, img_numpy):
        # 预处理: 转 Tensor, 归一化 0-1
        img_tensor = torch.from_numpy(img_numpy).float().unsqueeze(0).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            recon, z, _, _ = self.model(img_tensor)
        recon_numpy = recon.cpu().squeeze().numpy()
        # 返回列表，保持与 VAE 接口兼容 (index 2 是 recon)
        return [None, None, recon_numpy, z]


class Evaluator:
    def __init__(self, depths_folder, colls_folder, image_size=(480, 270),
                 latent_dims=[32, 64, 128, 256],
                 model_types=['Beta_VAE', 'DC_VAE', 'AE'],
                 batch_size=8):

        # image_size 格式为 (height, width)
        self.image_size = (image_size[1], image_size[0]) if len(image_size) == 2 else image_size
        self.latent_dims = latent_dims
        self.model_types = model_types
        self.batch_size = batch_size

        # 阈值参数
        # 2.0m ~= 73 (假设 7m = 255)
        self.danger_threshold = 73.0
        # 0.5m ~= 18
        self.safety_margin = 18.0

        self.min_depth = 15
        self.max_depth = 255

        # 数据集
        self.dataset = PureDepthCollisionDataset(
            depths_folder=depths_folder,
            colls_folder=colls_folder,
            transform=None,
            return_file_name=False
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=self._seed_worker  # 确保 DataLoader worker 的随机性也是固定的
        )

        # Beta-VAE 后处理需要的工具
        self.cam_params = CamParams(cx=240, cy=135, fx=252.91646, fy=252.91646)
        self.edge_detector = EdgeDetector(threshold1=30, threshold2=50)

        # 预加载模型
        self.models = self._preload_models()

    def _seed_worker(self, worker_id):
        """为 DataLoader worker 设置随机种子"""
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _preload_models(self):
        models = {}
        print("正在预加载所有模型...")
        base_path = "/home/niu/workspaces/VAE_ws/agent_encoder/weights"

        for model_type in self.model_types:
            models[model_type] = {}
            for ld in self.latent_dims:
                try:
                    reconstructor = self._create_reconstructor(model_type, ld, base_path)
                    models[model_type][ld] = reconstructor
                    print(f"已加载 {model_type} LD={ld}")
                except Exception as e:
                    # 打印完整的错误堆栈以便调试
                    import traceback
                    traceback.print_exc()
                    print(f"加载失败 {model_type} LD={ld}: {e}")
        return models

    def _create_reconstructor(self, model_type, latent_dim, base_path):
        if model_type == 'AE':
            model_path = os.path.join(base_path, f"dc_ae_LD_{latent_dim}_epoch_20.pth")
            # DepthAEReconstructor 是在当前文件中定义的，位置参数是确定的 (model_path, latent_dim, image_size)
            return DepthAEReconstructor(model_path, latent_dim, self.image_size)

        elif model_type == 'DC_VAE':
            # 使用 beta=3.0 的 DC-VAE
            model_path = os.path.join(base_path, f"dc_vae_beta3.0_LD_{latent_dim}_epoch_30.pth")
            # 修复：使用关键字参数 image_size=self.image_size
            return DepthVAEReconstructor(
                model_path=model_path,
                latent_dim=latent_dim,
                image_size=self.image_size,
                inference_mode=True
            )

        elif model_type == 'Beta_VAE':
            # 使用 beta=100.0 的 Beta-VAE
            model_path = os.path.join(base_path, f"beta_vae_beta100.0_LD_{latent_dim}_epoch_30.pth")
            # 修复：使用关键字参数 image_size=self.image_size
            return DepthVAEReconstructor(
                model_path=model_path,
                latent_dim=latent_dim,
                image_size=self.image_size,
                inference_mode=True
            )

        else:
            raise ValueError(f"未知的模型类型: {model_type}")

    def preprocess_depth_batch(self, depth_batch):
        batch_size = depth_batch.shape[0]
        processed_batch = np.zeros((batch_size, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for i in range(batch_size):
            depth_np = depth_batch[i].numpy().squeeze().astype(np.float32)
            if np.max(depth_np) > 255:
                depth_np = depth_np * 255 / 7000
            depth_np[depth_np > self.max_depth] = self.max_depth
            depth_np[depth_np < self.min_depth] = 0
            resized = cv2.resize(depth_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            processed_batch[i] = resized
        return processed_batch

    def preprocess_collision_batch(self, coll_batch):
        batch_size = coll_batch.shape[0]
        processed_batch = np.zeros((batch_size, self.image_size[0], self.image_size[1]), dtype=np.float32)
        for i in range(batch_size):
            coll_np = coll_batch[i].numpy().squeeze().astype(np.float32)
            resized = cv2.resize(coll_np, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
            processed_batch[i] = resized
        return processed_batch

    def clean_image(self, img):
        """
        清理图像数据：处理 NaN, Inf, 并截断到 0-255
        适用于 Batch (N, H, W) 或 Single (H, W)
        """
        img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(img, 0, 255)

    def process_beta_vae_output(self, recon_depth):
        """将Beta-VAE重构的深度图转换为碰撞图"""
        # 确保输入是 0-255 float32
        depth_img = self.clean_image(recon_depth)

        # 边缘检测 (需要 uint8)
        edges, _ = self.edge_detector.process_image(depth_img.astype(np.uint8))

        # Raycasting 生成碰撞图
        raycast_img, offset_depth = process_depth_image(depth_img, edges, self.cam_params)

        # 合成最终碰撞图
        coll_img = 25.5 * np.minimum(offset_depth, raycast_img)
        return self.clean_image(coll_img)

    def evaluate_batch(self, depth_batch, model_type, latent_dim):
        processed_depth = self.preprocess_depth_batch(depth_batch)
        reconstructor = self.models[model_type][latent_dim]

        predictions = []

        # 逐个样本推理 (因为 VAE 接口可能是单样本或多样本，且 Beta-VAE 后处理较慢)
        for i in range(len(processed_depth)):
            single_input = processed_depth[i]  # (H, W)

            # forward
            ret = reconstructor.forward(single_input)
            recon_raw = ret[2]  # 0-1 scale

            if isinstance(recon_raw, torch.Tensor):
                recon_raw = recon_raw.cpu().numpy()

            # 转为 0-255
            recon_img = recon_raw * 255.0

            if model_type == 'Beta_VAE':
                # Beta-VAE 输出的是深度图，需要转换为碰撞图
                final_coll = self.process_beta_vae_output(recon_img)
            else:
                # DC-VAE 和 AE 直接输出碰撞图
                final_coll = self.clean_image(recon_img)

            predictions.append(final_coll)

        return np.stack(predictions)

    # --- 单图像处理 (用于绘图) ---

    def preprocess_depth_single(self, image):
        if np.max(image) > 255: image = image * 255 / 7000
        image[image > self.max_depth] = self.max_depth
        image[image < self.min_depth] = 0
        return cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

    def preprocess_collision_single(self, image):
        return cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)

    def evaluate_model_single(self, depth_image, model_type, latent_dim):
        """对单张图像进行评估，返回重构的碰撞图"""
        processed_depth = self.preprocess_depth_single(depth_image)
        reconstructor = self.models[model_type][latent_dim]

        try:
            ret = reconstructor.forward(processed_depth)
            recon_raw = ret[2]  # 0-1 scale

            if isinstance(recon_raw, torch.Tensor):
                recon_raw = recon_raw.cpu().numpy()

            recon_img = recon_raw * 255.0

            if model_type == 'Beta_VAE':
                final_coll = self.process_beta_vae_output(recon_img)
            else:
                final_coll = self.clean_image(recon_img)

            return final_coll

        except Exception as e:
            print(f"{model_type} LD={latent_dim} 单图像处理错误: {e}")
            return np.zeros((self.image_size[0], self.image_size[1]), dtype=np.float32)

    # --- 指标计算 ---

    def calc_rmse(self, gt, pred):
        return np.sqrt(np.mean((gt - pred) ** 2))

    def calc_ssim(self, img1, img2):
        """计算结构相似性指数 (推荐版)"""
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        if np.isnan(img1).any() or np.isnan(img2).any(): return np.nan

        # 动态范围设定 (0-255)
        data_range = 255.0

        # 窗口大小自适应
        win_size = min(7, min(img1.shape) - 1)
        if win_size % 2 == 0: win_size -= 1
        win_size = max(3, win_size)

        try:
            ssim_value = ssim(img1, img2, data_range=data_range, win_size=win_size)
        except ValueError:
            ssim_value = np.nan
        return ssim_value

    def calc_c_rmse(self, gt, pred):
        """Critical-RMSE: 只在 GT < danger_threshold 的区域计算 RMSE"""
        mask = gt < self.danger_threshold
        if np.sum(mask) == 0:
            return np.nan  # 没有危险区域
        diff = (gt[mask] - pred[mask]) ** 2
        return np.sqrt(np.mean(diff))

    def calc_svr(self, gt, pred):
        """Safety Violation Rate: 预测值比真值大(更安全)超过 margin 的像素比例"""
        violation_mask = pred > (gt + self.safety_margin)
        return np.mean(violation_mask) * 100.0  # 返回百分比

    def run_evaluation(self, max_samples=None):
        results = {
            mt: {ld: {'rmse': [], 'ssim': [], 'c_rmse': [], 'svr': []} for ld in self.latent_dims}
            for mt in self.model_types
        }

        total_samples = len(self.dataset)
        if max_samples:
            total_samples = min(total_samples, max_samples)

        print(f"开始评估，总样本数限制: {max_samples if max_samples else 'All'}")

        for mt in self.model_types:
            for ld in self.latent_dims:
                if ld not in self.models[mt]:
                    continue

                print(f"\n正在评估: {mt} - LD {ld}")
                sample_count = 0

                for depth_batch, coll_batch in tqdm(self.dataloader, desc=f"{mt}-{ld}"):
                    if sample_count >= total_samples:
                        break

                    batch_preds = self.evaluate_batch(depth_batch, mt, ld)
                    batch_gts = self.preprocess_collision_batch(coll_batch)

                    for i in range(len(batch_preds)):
                        if sample_count >= total_samples:
                            break

                        gt = batch_gts[i]
                        pred = batch_preds[i]

                        rmse = self.calc_rmse(gt, pred)
                        _ssim = self.calc_ssim(gt, pred)
                        c_rmse = self.calc_c_rmse(gt, pred)
                        svr = self.calc_svr(gt, pred)

                        results[mt][ld]['rmse'].append(rmse)
                        results[mt][ld]['ssim'].append(_ssim)
                        if not np.isnan(c_rmse):
                            results[mt][ld]['c_rmse'].append(c_rmse)
                        results[mt][ld]['svr'].append(svr)

                        sample_count += 1

        final_stats = {}
        for mt in self.model_types:
            final_stats[mt] = {}
            for ld in self.latent_dims:
                if ld not in results[mt]: continue

                metrics = results[mt][ld]
                final_stats[mt][ld] = {
                    'rmse': np.nanmean(metrics['rmse']),
                    'ssim': np.nanmean(metrics['ssim']),
                    'c_rmse': np.nanmean(metrics['c_rmse']),
                    'svr': np.nanmean(metrics['svr']),
                    'count': len(metrics['rmse'])
                }

        return final_stats

    def print_table(self, stats):
        """
        格式化打印评估结果，采用按指标分组的 Markdown 表格格式
        """
        # 1. 定义配置
        # 列表元组: (metric_key, display_name, format_string)
        metrics_config = [
            ('rmse', 'RMSE ↓', '{:.4f}'),
            ('ssim', 'SSIM ↑', '{:.4f}'),
            ('c_rmse', 'C-RMSE ↓', '{:.4f}'),
            ('svr', 'SVR(%) ↓', '{:.2f}')
        ]

        model_display_names = {
            'Beta_VAE': '- 级联 $\\beta$-VAE',
            'AE': '- AE',
            'DC_VAE': '- DC-VAE'
        }

        # 指定输出模型的顺序
        ordered_models = ['Beta_VAE', 'AE', 'DC_VAE']

        # 2. 构建表头
        col_width = 12
        first_col_width = 22

        # Header Row: | **方法 \ $h_{dim}$** | **32** | **64** | ... |
        header_cells = [f" {ld}".ljust(col_width) for ld in self.latent_dims]
        header_row = f"| {'方法 - $h_{dim}$'.ljust(first_col_width)} | " + " | ".join(header_cells) + " |"

        # Separator Row: | --- | --- | ... |
        sep_cells = ["-" * col_width for _ in self.latent_dims]
        sep_row = f"| {'-' * first_col_width} | " + " | ".join(sep_cells) + " |"

        output_lines = [header_row, sep_row]

        # 3. 构建数据行
        for metric_key, metric_label, fmt in metrics_config:
            # 每个指标的标题行，后面留空
            metric_header = f"| {metric_label.ljust(first_col_width)} | " + " | ".join(
                [" " * col_width for _ in self.latent_dims]) + " |"
            output_lines.append(metric_header)

            for mt in ordered_models:
                # 如果该模型没有被评估，跳过
                if mt not in stats: continue

                display_name = model_display_names.get(mt, f"- {mt}")
                row_cells = []

                for ld in self.latent_dims:
                    if ld in stats[mt]:
                        val = stats[mt][ld][metric_key]
                        val_str = fmt.format(val)
                        row_cells.append(val_str.ljust(col_width))
                    else:
                        row_cells.append("N/A".ljust(col_width))

                row_str = f"| {display_name.ljust(first_col_width)} | " + " | ".join(row_cells) + " |"
                output_lines.append(row_str)

        # 4. 输出与保存
        final_output = "\n".join(output_lines)
        print("\n" + final_output)

        with open("comprehensive_evaluation_result4.txt", "w") as f:
            f.write(final_output)
        print("结果已保存至 comprehensive_evaluation_result4.txt")

    def plot_three_model_comparison(self, image_index=0, save_path="Figure3-Y_Comparison.png"):
        """
        绘制三种模型 (Cascade Beta-VAE, AE, DC-VAE) 在不同潜在维度下的重构对比图
        对应论文中的图 3-Y
        """
        def_font_size = 40
        depth, coll = self.dataset[image_index]
        depth = depth.squeeze().astype(np.float32)
        coll = coll.squeeze().astype(np.float32)

        orig_depth = self.preprocess_depth_single(depth)
        orig_coll = self.preprocess_collision_single(coll)

        # 定义要绘制的模型顺序和标签
        rows_config = [
            {'model': 'Beta_VAE', 'label': r'$\beta$-VAE'},
            {'model': 'AE', 'label': 'AE'},
            {'model': 'DC_VAE', 'label': 'DC-VAE'}
        ]

        n_rows = len(rows_config)
        n_cols = len(self.latent_dims) + 2  # +2 for Orig Depth and Orig Coll

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

        # 标题列
        # col_titles = ["Depth Input", "Collision GT"] + [f"Recon (LD={ld})" for ld in self.latent_dims]
        col_titles = ["原始深度图", "目标碰撞图"] + [f"LD={ld}" for ld in self.latent_dims]
        for ax, col_title in zip(axes[0], col_titles):
            ax.set_title(col_title, fontsize=def_font_size, fontweight='bold', pad=15)

        for i, config in enumerate(rows_config):
            ax_row = axes[i]
            model_type = config['model']
            label = config['label']

            # 设置行标签
            ax_row[0].set_ylabel(label, fontsize=def_font_size, fontweight='bold', rotation=0,
                                 labelpad=60, verticalalignment='center')

            # 第一列: 原始深度图 (所有行都显示，作为参考)
            ax_row[0].imshow(orig_depth, cmap='jet')

            # 第二列: 原始碰撞图 GT
            ax_row[1].imshow(orig_coll, cmap='jet')

            # 后续列: 各个 LD 下的模型重构结果
            for col_idx, ld in enumerate(self.latent_dims, start=2):
                if ld in self.models[model_type]:
                    final_coll = self.evaluate_model_single(depth, model_type, ld)
                    ax_row[col_idx].imshow(final_coll, cmap='jet')
                else:
                    ax_row[col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')

        # 移除坐标轴刻度
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                # 移除边框
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close(fig)
        print(f"对比图已保存至: {save_path}")

    def plot_two_model_comparison(self, image_index=0, save_path="Figure3-Y_AE_DC_Comparison.png"):
        """
        只绘制 AE 和 DC-VAE 的对比图 (用于论文插图)
        """
        # 仅选择这两个模型进行对比
        comparison_models = ['AE', 'DC_VAE']

        depth, coll = self.dataset[image_index]
        depth = depth.squeeze().astype(np.float32)
        coll = coll.squeeze().astype(np.float32)

        orig_depth = self.preprocess_depth_single(depth)
        orig_coll = self.preprocess_collision_single(coll)

        # 定义绘图配置
        rows_config = [
            {'model': 'AE', 'label': 'AE'},
            {'model': 'DC_VAE', 'label': 'DC-VAE'}
        ]

        n_rows = len(rows_config)
        n_cols = len(self.latent_dims) + 2  # +2 for Depth Input and Collision GT

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

        # 如果只有一行，axes可能是1D数组，强制转2D
        if n_rows == 1: axes = np.array([axes])

        # 标题列
        col_titles = ["Depth Input", "Collision GT"] + [f"Recon (LD={ld})" for ld in self.latent_dims]
        for ax, col_title in zip(axes[0], col_titles):
            ax.set_title(col_title, fontsize=18, fontweight='bold', pad=15)

        for i, config in enumerate(rows_config):
            ax_row = axes[i]
            model_type = config['model']
            label = config['label']

            # 设置行标签
            ax_row[0].set_ylabel(label, fontsize=16, fontweight='bold', rotation=0, labelpad=60)

            # 第一列: 原始深度图
            ax_row[0].imshow(orig_depth, cmap='jet')

            # 第二列: 原始碰撞图 GT
            ax_row[1].imshow(orig_coll, cmap='jet')

            # 后续列: 各个 LD 下的模型重构结果
            for col_idx, ld in enumerate(self.latent_dims, start=2):
                # 检查模型是否已加载
                if model_type in self.models and ld in self.models[model_type]:
                    final_coll = self.evaluate_model_single(depth, model_type, ld)
                    ax_row[col_idx].imshow(final_coll, cmap='jet')
                else:
                    ax_row[col_idx].text(0.5, 0.5, 'N/A', ha='center', va='center')
                    ax_row[col_idx].set_facecolor('#f0f0f0')  # 灰色背景表示缺失

        # 移除坐标轴刻度
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                # 移除边框
                for spine in ax.spines.values():
                    spine.set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"AE与DC-VAE对比图已保存至: {save_path}")


if __name__ == "__main__":
    # 设置全局随机种子
    set_seed(42)

    # 路径配置
    # DEPTHS_FOLDER = "/home/niu/workspaces/VAE_ws/data_test/depths"  # 测试用测试集
    # COLLS_FOLDER = "/home/niu/workspaces/VAE_ws/data_test/colls"    #
    DEPTHS_FOLDER = "/home/niu/workspaces/VAE_ws/datasets/depths"   # 绘图可用训练集
    COLLS_FOLDER = "/home/niu/workspaces/VAE_ws/datasets/colls_target"

    evaluator = Evaluator(
        depths_folder=DEPTHS_FOLDER,
        colls_folder=COLLS_FOLDER,
        latent_dims=[32, 64, 128, 256],
        model_types=['Beta_VAE','AE', 'DC_VAE'],  # 对比三种模型'Beta_VAE',   'Beta_VAE',
        batch_size=32
    )

    # 1. 运行评估 (max_samples=None 跑全量, 或者设置数值快速测试)
    # stats = evaluator.run_evaluation(max_samples=None)
    # evaluator.print_table(stats)

    # 2. 绘制论文图 3-Y
    # 选择一个具有挑战性的样本索引 (例如包含细小障碍物或边缘的场景)
    evaluator.plot_three_model_comparison(image_index=49994, save_path="2.三种VAE方法对比.png")

    # for idx in range(49950, 50000):  # 示例索引  寻找DC-VAE和AE的效果差异时可以用这个
    #     evaluator.plot_two_model_comparison(image_index=idx, save_path=f"figs/两种VAE方法对比{idx}.png")