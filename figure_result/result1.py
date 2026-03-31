import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
from compare_methods.fft import FFT
from compare_methods.wavelets import WaveletTransforms
from agent_encoder.vae_image_a_test import DepthVAEReconstructor
from agent_encoder.Display_pure_depth_dataset import DepthPureDataset
import warnings
from matplotlib import font_manager

# 手动指定字体路径
font_path = '/usr/share/fonts/MyFonts/simhei.ttf'
font_prop = font_manager.FontProperties(fname=font_path)

# 设置字体
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


class Evaluator:
    def __init__(self, dataset_path, image_size=(480, 270), latent_dim=64):
        self.image_size = image_size
        self.min_depth = 15
        self.max_depth = 255

        # 初始化三种方法
        self.fft_processor = FFT()
        self.wavelet_processor = WaveletTransforms(name='db1', level=6)
        self.vae_reconstructor = DepthVAEReconstructor(
            model_path=f"/home/niu/workspaces/VAE_ws/agent_encoder/weights/old_weights_with_less_data_augement/"
                       f"990_beta_3_LD_{latent_dim}_epoch_40.pth",
            latent_dim=latent_dim,
            image_size=image_size
        )

        # 创建数据集
        self.dataset = DepthPureDataset(
            depths_folder=dataset_path,
            transform=None,
            return_file_name=False
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4
        )

    def _get_vae_reconstructor(self, latent_dim):
        """根据潜在维度创建VAE重建器实例"""
        return DepthVAEReconstructor(
            model_path=f"/home/niu/workspaces/VAE_ws/agent_encoder/weights/old_weights_with_less_data_augement/"
                       f"990_beta_3_LD_{latent_dim}_epoch_40.pth",
            latent_dim=latent_dim,
            image_size=self.image_size
        )

    def preprocess_image(self, image):
        """统一预处理图像"""
        if np.max(image) > 255:
            image = image * 255 / 7000  # DIML/CVl数据集特定缩放

        image[image > self.max_depth] = self.max_depth
        image[image < self.min_depth] = 0
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)

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

    def clean_image(self, image):
        """清理图像中的无效值"""
        if np.isnan(image).any():
            image = np.nan_to_num(image, nan=0.0)

        if np.isinf(image).any():
            max_val = np.max(image[np.isfinite(image)])
            if np.isnan(max_val) or np.isinf(max_val):
                max_val = 255.0
            image = np.nan_to_num(image, posinf=max_val, neginf=0.0)

        image = np.clip(image, 0, 255)
        return image

    def evaluate_fft(self, image, latent_dims=128):
        """FFT方法评估"""
        processed = self.preprocess_image(image)
        try:
            _, reconstructed = self.fft_processor.get_image_reconstruction_with_compressed_dimensions(
                processed, latent_dims
            )
            reconstructed = np.clip(reconstructed, 0, 255)
            return processed, self.clean_image(reconstructed)
        except Exception as e:
            print(f"FFT处理错误: {e}")
            return processed, np.zeros_like(processed)

    def evaluate_wavelet(self, image, latent_dims=128):
        """小波方法评估"""
        processed = self.preprocess_image(image)
        try:
            _, reconstructed = self.wavelet_processor.get_wavelet_coefficients(
                processed, latent_dims
            )
            if reconstructed.shape != processed.shape:
                reconstructed = cv2.resize(reconstructed, (processed.shape[1], processed.shape[0]))
            reconstructed = np.clip(reconstructed, 0, 255)
            return processed, self.clean_image(reconstructed)
        except Exception as e:
            print(f"小波处理错误: {e}")
            return processed, np.zeros_like(processed)

    def evaluate_vae(self, image, latent_dim=None):
        """VAE方法评估 - 已修复"""
        processed = self.preprocess_image(image)
        try:
            # 如果指定了潜在维度，则使用新的重建器
            if latent_dim is not None:
                reconstructor = self._get_vae_reconstructor(latent_dim)
                # 修正：正确接收三个返回值
                depth_orig, _, reconstructed = reconstructor.forward(processed)
            else:
                depth_orig, _, reconstructed = self.vae_reconstructor.forward(processed)

            reconstructed = (reconstructed * 255).astype(np.float32)
            return processed, self.clean_image(reconstructed)
        except Exception as e:
            print(f"VAE处理错误: {e}")
            return processed, np.zeros_like(processed)

    def run_evaluation(self, latent_dims=128, max_samples=None, methods=['fft', 'wavelet', 'vae']):
        """运行完整评估"""
        results = {
            'fft': {'rmse': [], 'ssim': []},
            'wavelet': {'rmse': [], 'ssim': []},
            'vae': {'rmse': [], 'ssim': []}
        }

        total_samples = len(self.dataloader)
        if max_samples and max_samples < total_samples:
            total_samples = max_samples

        iterator = tqdm(enumerate(self.dataloader), total=total_samples, desc="Evaluating")
        for i, (depth, _) in iterator:
            if i >= total_samples:
                break

            depth = depth.numpy().squeeze().astype(np.float32)

            rmse_fft, ssim_fft = 0, 0
            rmse_wave, ssim_wave = 0, 0
            rmse_vae, ssim_vae = 0, 0

            if 'fft' in methods:
                try:
                    orig_fft, recon_fft = self.evaluate_fft(depth, latent_dims)
                    rmse_fft = self.calculate_rmse(orig_fft, recon_fft)
                    ssim_fft = self.calculate_ssim(orig_fft, recon_fft)
                except Exception as e:
                    print(f"FFT评估出错: {e}")
                    rmse_fft, ssim_fft = np.nan, np.nan
                results['fft']['rmse'].append(rmse_fft)
                results['fft']['ssim'].append(ssim_fft)

            if 'wavelet' in methods:
                try:
                    orig_wave, recon_wave = self.evaluate_wavelet(depth, latent_dims)
                    rmse_wave = self.calculate_rmse(orig_wave, recon_wave)
                    ssim_wave = self.calculate_ssim(orig_wave, recon_wave)
                except Exception as e:
                    print(f"小波评估出错: {e}")
                    rmse_wave, ssim_wave = np.nan, np.nan
                results['wavelet']['rmse'].append(rmse_wave)
                results['wavelet']['ssim'].append(ssim_wave)

            if 'vae' in methods:
                try:
                    orig_vae, recon_vae = self.evaluate_vae(depth)
                    rmse_vae = self.calculate_rmse(orig_vae, recon_vae)
                    ssim_vae = self.calculate_ssim(orig_vae, recon_vae)
                except Exception as e:
                    print(f"VAE评估出错: {e}")
                    rmse_vae, ssim_vae = np.nan, np.nan
                results['vae']['rmse'].append(rmse_vae)
                results['vae']['ssim'].append(ssim_vae)

            postfix = {}
            if 'fft' in methods:
                postfix['FFT_RMSE'] = f"{rmse_fft:.2f}" if not np.isnan(rmse_fft) else "NaN"
            if 'wavelet' in methods:
                postfix['Wavelet_RMSE'] = f"{rmse_wave:.2f}" if not np.isnan(rmse_wave) else "NaN"
            if 'vae' in methods:
                postfix['VAE_RMSE'] = f"{rmse_vae:.2f}" if not np.isnan(rmse_vae) else "NaN"

            iterator.set_postfix(postfix)

        for method in methods:
            valid_rmse = [x for x in results[method]['rmse'] if not np.isnan(x)]
            valid_ssim = [x for x in results[method]['ssim'] if not np.isnan(x)]

            if valid_rmse:
                results[method]['avg_rmse'] = np.mean(valid_rmse)
            else:
                results[method]['avg_rmse'] = np.nan

            if valid_ssim:
                results[method]['avg_ssim'] = np.mean(valid_ssim)
            else:
                results[method]['avg_ssim'] = np.nan

        for method in ['fft', 'wavelet', 'vae']:
            if method not in methods:
                results[method]['avg_rmse'] = 0
                results[method]['avg_ssim'] = 0

        return results

    def print_results(self, results):
        """打印评估结果"""
        print("\n" + "=" * 50)
        print("深度图重建方法性能评估")
        print("=" * 50)
        print(f"{'Method':<10} |  {'Avg RMSE ↓':<10} | {'Avg SSIM ↑':<10} | {'Samples':<10}")
        print("-" * 50)

        for method, metrics in results.items():
            rmse = metrics['avg_rmse']
            ssim_val = metrics['avg_ssim']

            rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "NaN"
            ssim_str = f"{ssim_val:.4f}" if not np.isnan(ssim_val) else "NaN"

            print(f"{method:<10} | {rmse_str:<10} | {ssim_str:<10} | {len(metrics['rmse']):<10}")

        print("=" * 50)

        with open("evaluation_results.txt", "w") as f:
            f.write("深度图重建方法性能评估\n")
            f.write("=" * 50 + "\n")
            f.write(f"{'Method':<10} | {'Avg RMSE ↓':<10} | {'Avg SSIM ↑':<10} | {'Samples':<10}\n")
            f.write("-" * 50 + "\n")

            for method, metrics in results.items():
                rmse = metrics['avg_rmse']
                ssim_val = metrics['avg_ssim']

                rmse_str = f"{rmse:.4f}" if not np.isnan(rmse) else "NaN"
                ssim_str = f"{ssim_val:.4f}" if not np.isnan(ssim_val) else "NaN"

                f.write(f"{method:<10} | {rmse_str:<10} | {ssim_str:<10} | {len(metrics['rmse']):<10}\n")

            f.write("=" * 50 + "\n")

        print("结果已保存到 evaluation_results.txt")

    def plot_comparison(self, image_index=0, save_path="comparison_plot.png"):
        """
        绘制不同方法和潜在维度下的深度图比较（包含差异图）
        修改：增大了标签和标题的字体大小，并调整了布局以适应大字体。
        """
        depth, _ = self.dataset[image_index]
        depth = depth.squeeze().astype(np.float32)

        original = self.preprocess_image(depth)

        latent_dims = [32, 64, 128, 256]

        # 1. 定义字体大小配置（在此处统一修改）
        TITLE_SIZE = 30  # 列标题字体大小 (原16)
        ROW_LABEL_SIZE = 30  # 行标签字体大小 (原16)
        MAIN_LABEL_SIZE = 30  # "潜在维度"标签大小 (原16)

        fig, axes = plt.subplots(nrows=4, ncols=7, figsize=(24, 8))

        # 2. 调整布局参数
        # left从0.05调整为0.06，给左侧大字体留出更多空间
        # top从0.99调整为0.92，给顶部大标题留出更多空间
        plt.subplots_adjust(
            wspace=0.01,
            hspace=0.01,
            left=0.04,  # 增加左边距
            right=1,
            top=0.92,  # 增加上边距
            bottom=0.00
        )

        # 3. 调整 "潜在维度" 总标签的位置和大小
        # x坐标从0.035左移至0.01，防止遮挡；y坐标稍微下移适配布局
        fig.text(0.01, 0.94, '$h_{dim}$', fontsize=MAIN_LABEL_SIZE, fontweight='bold')

        col_titles = [
            '原始深度图',
            'VAE重建',
            'FFT重建',
            '小波重建',
            'VAE误差',
            'FFT误差',
            '小波误差'
        ]

        # 4. 设置列标题
        for col_idx, col_title in enumerate(col_titles):
            # pad从5增加到12，防止大字压到图片
            axes[0, col_idx].set_title(col_title, fontsize=TITLE_SIZE, fontweight='bold', pad=10)

        global_max_error = 0

        all_reconstructions = {}
        all_errors = {}

        row_height = 0.23
        base_y = 0.8

        for row_idx, ld in enumerate(latent_dims):
            axes[row_idx, 0].imshow(original, cmap='jet')

            # 5. 设置行标签 (32, 64...)
            # x坐标从0.04左移至0.03，适配大字体
            fig.text(0.02, base_y - row_idx * row_height, str(ld),
                     fontsize=ROW_LABEL_SIZE, ha='center', va='center', fontweight='bold')

            # VAE重建
            _, vae_recon = self.evaluate_vae(depth, latent_dim=ld)
            axes[row_idx, 1].imshow(vae_recon, cmap='jet')
            vae_error = np.abs(original - vae_recon)
            all_reconstructions[(row_idx, 'vae')] = vae_recon
            all_errors[(row_idx, 'vae')] = vae_error
            global_max_error = max(global_max_error, np.max(vae_error))

            # FFT重建
            _, fft_recon = self.evaluate_fft(depth, latent_dims=ld)
            axes[row_idx, 2].imshow(fft_recon, cmap='jet')
            fft_error = np.abs(original - fft_recon)
            all_reconstructions[(row_idx, 'fft')] = fft_recon
            all_errors[(row_idx, 'fft')] = fft_error
            global_max_error = max(global_max_error, np.max(fft_error))

            # 小波重建
            _, wavelet_recon = self.evaluate_wavelet(depth, latent_dims=ld)
            axes[row_idx, 3].imshow(wavelet_recon, cmap='jet')
            wavelet_error = np.abs(original - wavelet_recon)
            all_reconstructions[(row_idx, 'wavelet')] = wavelet_recon
            all_errors[(row_idx, 'wavelet')] = wavelet_error
            global_max_error = max(global_max_error, np.max(wavelet_error))

        for row_idx, ld in enumerate(latent_dims):
            vae_error = all_errors[(row_idx, 'vae')]
            im_vae_error = axes[row_idx, 4].imshow(vae_error, vmin=0, vmax=global_max_error)

            fft_error = all_errors[(row_idx, 'fft')]
            im_fft_error = axes[row_idx, 5].imshow(fft_error, vmin=0, vmax=global_max_error)

            wavelet_error = all_errors[(row_idx, 'wavelet')]
            im_wavelet_error = axes[row_idx, 6].imshow(wavelet_error, vmin=0, vmax=global_max_error)

        for ax_row in axes:
            for ax in ax_row:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

        # 注意：bbox_inches='tight' 会自动裁剪空白，有时会切掉使用 fig.text 添加的边缘文字
        # 如果发现文字被切掉，可以尝试移除 bbox_inches='tight' 或者增加 pad_inches
        plt.savefig(save_path, bbox_inches='tight', dpi=500)
        plt.close(fig)
        print(f"包含差异图的比较图已保存至: {save_path}")


if __name__ == "__main__":
    DATASET_PATH = "/home/niu/workspaces/VAE_ws/data_test/depths"
    LATENT_DIMS = 64

    evaluator = Evaluator(DATASET_PATH, latent_dim=LATENT_DIMS)

    # 运行评估（可选）
    # results = evaluator.run_evaluation(LATENT_DIMS, methods=['fft', 'wavelet', 'vae'])
    # evaluator.print_results(results)

    # 绘制比较图
    evaluator.plot_comparison(image_index=105, save_path="2.fft等方法对比.png")