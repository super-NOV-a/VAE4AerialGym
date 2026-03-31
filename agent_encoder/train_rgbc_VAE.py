import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from Project_dataset_show import RGBDepthCollisionDataset, preprocess_image, preprocess_rgb_image
from agent_encoder.Net.VAE import VAE, ImgEncoder  # 导入VAE和ImgEncoder


# ==============================================================
# RGB Encoder (与深度编码器结构相同，但输入通道为3)
# ==============================================================

class RGBEncoder(nn.Module):
    """
    RGB编码器，结构与深度编码器相同，但输入通道为3
    """

    def __init__(self, input_dim=3, latent_dim=64):
        super(RGBEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.elu = nn.ELU()
        print("Defined RGB encoder.")

    def define_encoder(self):
        # 与深度编码器相同的结构
        self.conv0 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=2, padding=2)
        self.conv0_1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2)
        nn.init.xavier_uniform_(self.conv0_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv0_1.bias)

        self.conv1_0 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv1_1.bias)

        self.conv2_0 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        nn.init.xavier_uniform_(self.conv2_1.weight, gain=nn.init.calculate_gain("linear"))
        nn.init.zeros_(self.conv2_1.bias)

        self.conv3_0 = nn.Conv2d(128, 128, kernel_size=5, stride=2)

        self.conv0_jump_2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv1_jump_3 = nn.Conv2d(64, 128, kernel_size=5, stride=4, padding=(2, 1))

        self.dense0 = nn.Linear(3 * 6 * 128, 512)
        self.dense1 = nn.Linear(512, 2 * self.latent_dim)

    def forward(self, img):
        return self.encode(img)

    def encode(self, img):
        """
        Encodes the input RGB image.
        """
        # conv0
        x0_0 = self.conv0(img)
        x0_1 = self.conv0_1(x0_0)
        x0_1 = self.elu(x0_1)

        x1_0 = self.conv1_0(x0_1)
        x1_1 = self.conv1_1(x1_0)

        x0_jump_2 = self.conv0_jump_2(x0_1)

        x1_1 = x1_1 + x0_jump_2
        x1_1 = self.elu(x1_1)

        x2_0 = self.conv2_0(x1_1)
        x2_1 = self.conv2_1(x2_0)

        x1_jump3 = self.conv1_jump_3(x1_1)

        x2_1 = x2_1 + x1_jump3
        x2_1 = self.elu(x2_1)

        x3_0 = self.conv3_0(x2_1)

        x = x3_0.view(x3_0.size(0), -1)
        x = self.dense0(x)
        x = self.elu(x)
        x = self.dense1(x)
        return x


# ==============================================================
# 损失函数 - 包含重构损失、KL散度和特征对齐损失
# ==============================================================

class RGBVAELoss(nn.Module):
    def __init__(self, beta_coeff=0.01, alpha_align=1.0, gamma_recon=1.0, latent_dims=64):
        super(RGBVAELoss, self).__init__()
        self.beta_coeff = beta_coeff
        self.alpha_align = alpha_align
        self.gamma_recon = gamma_recon
        self.latent_dims = latent_dims
        self.beta_eff = (beta_coeff * latent_dims) / (480.0 * 270.0)

    def forward(self, recon_x, x, rgb_mean, rgb_logvar, depth_mean):
        # 重构损失 - 修正版本（方案1）
        invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        mse_loss = nn.MSELoss(reduction="none")
        cross_ent = mse_loss(recon_x, x) * invalid_pixel_mask

        # 计算所有有效像素的平均MSE
        valid_pixel_count = torch.sum(invalid_pixel_mask)
        if valid_pixel_count > 0:
            reconstruction_loss = torch.sum(cross_ent) / valid_pixel_count
        else:
            reconstruction_loss = torch.tensor(0.0, device=x.device)

        # KL散度
        kld_loss = -0.5 * torch.mean(torch.sum(1 + rgb_logvar - rgb_mean.pow(2) - rgb_logvar.exp(), dim=1))

        # 特征对齐损失 - 使用MSE
        alignment_loss = F.mse_loss(rgb_mean, depth_mean)

        # # 总损失（只有对齐损失）
        # total_loss = self.alpha_align * alignment_loss
        # cosine_sim = F.cosine_similarity(rgb_mean, depth_mean, dim=1)
        # alignment_loss = torch.mean(1 - cosine_sim)

        # 总损失
        total_loss = (self.gamma_recon * reconstruction_loss +
                      self.beta_eff * kld_loss +
                      self.alpha_align * alignment_loss)

        return total_loss, reconstruction_loss, kld_loss, alignment_loss


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average KL per latent dimension across the batch (nats). Shape: [latent_dim]."""
    kld_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kld_elem.mean(dim=0)


# ==============================================================
# 训练函数
# ==============================================================

def train_rgb_vae(
        rgbs_folder: str,
        depths_folder: str,
        colls_folder: str,
        pretrained_vae_path: str,
        dc_beta: str,
        latent_dim: int = 64,
        beta_coeff: float = 3.0,
        alpha_align: float = 1.0,  # 特征对齐损失的权重
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_name: str = "rgbc_vae",
        seed: int = 42,
):
    # Reproducibility - 最严格的设置（与train_dc_VAE保持一致）
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建固定的生成器
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    # 加载数据集 (包含RGB) - 使用生成器和worker初始化
    dataset = RGBDepthCollisionDataset(
        depths_folder=depths_folder,
        colls_folder=colls_folder,
        rgbs_folder=rgbs_folder,
        transform=preprocess_image,
        rgb_transform=preprocess_rgb_image,
        include_rgb=True,  # 重要：包含RGB图像
        is_simulate=True,   # 只使用模拟数据
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker
    )

    # 加载预训练的深度VAE并冻结
    print("Loading pre-trained VAE...")
    pretrained_vae = VAE(input_dim=1, latent_dim=latent_dim).to(device)

    # 加载权重（需要适配您的权重加载逻辑）
    state_dict = torch.load(pretrained_vae_path)
    pretrained_vae.load_state_dict(state_dict)

    # 冻结预训练VAE的所有参数
    for param in pretrained_vae.parameters():
        param.requires_grad = False
    pretrained_vae.eval()
    print("Pre-trained VAE loaded and frozen.")

    # 创建RGB编码器
    rgb_encoder = RGBEncoder(input_dim=3, latent_dim=latent_dim).to(device)

    # 优化器只优化RGB编码器
    optimizer = optim.Adam(rgb_encoder.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 损失函数
    criterion = RGBVAELoss(beta_coeff=beta_coeff, alpha_align=alpha_align, latent_dims=latent_dim)

    # TensorBoard
    os.makedirs("weights_rgb", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{run_name}_dc{dc_beta}_beta{beta_coeff}_alpha{alpha_align}_LD_{latent_dim}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        rgb_encoder.train()
        epoch_total, epoch_rec, epoch_kl, epoch_align = 0.0, 0.0, 0.0, 0.0

        with tqdm(total=len(loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for rgb_batch, depth_batch, coll_batch in loader:
                rgb_batch = rgb_batch.to(device)
                depth_batch = depth_batch.to(device)
                coll_batch = coll_batch.to(device)

                # 使用冻结的深度编码器获取目标潜在特征
                with torch.no_grad():
                    depth_z = pretrained_vae.encoder(depth_batch)
                    depth_mean = depth_z[:, :latent_dim]  # 只取均值部分作为对齐目标

                # RGB编码器前向传播
                rgb_z = rgb_encoder(rgb_batch)
                rgb_mean = rgb_z[:, :latent_dim]
                rgb_logvar = rgb_z[:, latent_dim:]

                # 重参数化采样
                std = torch.exp(0.5 * rgb_logvar)
                eps = torch.randn_like(std)
                z_sampled = rgb_mean + eps * std

                # 使用冻结的解码器重构碰撞图
                with torch.no_grad():
                    recon_collision = pretrained_vae.img_decoder(z_sampled)

                # 计算损失
                total_loss, rec_loss, kl_loss, align_loss = criterion(
                    recon_collision, coll_batch, rgb_mean, rgb_logvar, depth_mean
                )

                # 反向传播
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rgb_encoder.parameters(), max_norm=1.0)
                optimizer.step()

                # 累积损失
                epoch_total += total_loss.item()
                epoch_rec += rec_loss.item()
                epoch_kl += kl_loss.item()
                epoch_align += align_loss.item()

                # 记录到TensorBoard
                writer.add_scalar("Loss/step_recon", rec_loss.item(), global_step)
                writer.add_scalar("Loss/step_kl", kl_loss.item(), global_step)
                writer.add_scalar("Loss/step_alignment", align_loss.item(), global_step)
                writer.add_scalar("Loss/step_total", total_loss.item(), global_step)
                writer.add_scalar("Meta/lr", optimizer.param_groups[0]["lr"], global_step)

                # KL per-dimension histogram
                kld_dim = kl_per_dim(rgb_mean, rgb_logvar).detach().cpu().numpy()
                writer.add_histogram("KL/per_dim", kld_dim, global_step)

                # 余弦相似度
                cosine_sim = F.cosine_similarity(rgb_mean, depth_mean, dim=1).mean().item()
                writer.add_scalar("Metrics/cosine_similarity", cosine_sim, global_step)

                pbar.set_postfix(
                    total=f"{total_loss.item():.4f}",
                    rec=f"{rec_loss.item():.4f}",
                    kl=f"{kl_loss.item():.4f}",
                    align=f"{align_loss.item():.4f}",
                    cos_sim=f"{cosine_sim:.4f}"
                )
                pbar.update(1)
                global_step += 1

        scheduler.step()

        # 计算epoch平均损失
        n_batches = len(loader)
        avg_total = epoch_total / n_batches
        avg_rec = epoch_rec / n_batches
        avg_kl = epoch_kl / n_batches
        avg_align = epoch_align / n_batches

        print(f"Epoch {epoch}/{epochs} | "
              f"Total: {avg_total:.4f} | Rec: {avg_rec:.4f} | "
              f"KL: {avg_kl:.4f} | Align: {avg_align:.4f}")

        writer.add_scalar("Loss/epoch_avg_total", avg_total, epoch)
        writer.add_scalar("Loss/epoch_avg_recon", avg_rec, epoch)
        writer.add_scalar("Loss/epoch_avg_kl", avg_kl, epoch)
        writer.add_scalar("Loss/epoch_avg_alignment", avg_align, epoch)

        # 可视化结果
        with torch.no_grad():
            # 选择第一个样本进行可视化
            idx = 0
            rgb_vis = rgb_batch[idx].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            depth_vis = depth_batch[idx].cpu().numpy()[0]  # 去掉通道维度
            coll_vis = coll_batch[idx].cpu().numpy()[0]
            recon_vis = recon_collision[idx].cpu().numpy()[0]

            # 创建可视化图像
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # 第一行：输入图像
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title("RGB Input")
            axes[0, 0].axis('off')

            axes[0, 1].imshow(depth_vis, cmap='gray', vmin=0, vmax=1)
            axes[0, 1].set_title("Depth Input")
            axes[0, 1].axis('off')

            axes[0, 2].imshow(coll_vis, cmap='gray', vmin=0, vmax=1)
            axes[0, 2].set_title("GT Collision")
            axes[0, 2].axis('off')

            # 第二行：重构结果和特征
            axes[1, 0].imshow(recon_vis, cmap='gray', vmin=0, vmax=1)
            axes[1, 0].set_title("Recon Collision")
            axes[1, 0].axis('off')

            # 特征分布可视化
            rgb_mean_np = rgb_mean[idx].cpu().numpy()
            depth_mean_np = depth_mean[idx].cpu().numpy()
            axes[1, 1].bar(range(len(rgb_mean_np)), rgb_mean_np, alpha=0.7, label='RGB')
            axes[1, 1].bar(range(len(depth_mean_np)), depth_mean_np, alpha=0.7, label='Depth')
            axes[1, 1].set_title("Latent Features")
            axes[1, 1].legend()

            # 余弦相似度热图
            sim_matrix = torch.matmul(rgb_mean, depth_mean.T).cpu().numpy()
            im = axes[1, 2].imshow(sim_matrix[:10, :10], cmap='viridis')  # 只显示前10个样本
            axes[1, 2].set_title("Feature Similarity Matrix")
            plt.colorbar(im, ax=axes[1, 2])

            plt.tight_layout()
            fig.canvas.draw()
            merged = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            merged = merged.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            merged_t = torch.from_numpy(merged).permute(2, 0, 1).float() / 255.0
            writer.add_image("Images/rgb_depth_recon_features", merged_t, epoch, dataformats="CHW")

        # 保存检查点
        if epoch % 10 == 0:
            ckpt = f"weights_rgb/rgb_vae_dc{dc_beta}_beta{beta_coeff}_alpha{alpha_align}_LD_{latent_dim}_epoch_{epoch}.pth"
            torch.save(rgb_encoder.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    writer.close()

    # # 保存最终模型
    # final_ckpt = f"weights_rgb/rgb_vae_dc{dc_beta}_beta{beta_coeff}_alpha{alpha_align}_LD_{latent_dim}_final.pth"
    # torch.save(rgb_encoder.state_dict(), final_ckpt)
    # print(f"Final model saved: {final_ckpt}")


if __name__ == "__main__":
    # 数据集路径
    rgbs_folder = "/home/niu/workspaces/VAE_ws/datasets/rgbs"
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_target"
    # dc_beta = str(3.0)  # 0.1, 0.3, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0
    dc_betas = "0.1,0.3,1.0,3.0,10.0,30.0,100.0,300.0,1000.0".split(",")

    for dc_beta in dc_betas:
        # 预训练VAE模型路径
        pretrained_vae_path = "/home/niu/workspaces/VAE_ws/agent_encoder/weights/dc_vae_beta"+dc_beta+"_LD_64_epoch_30.pth"

        # 训练参数
        train_rgb_vae(
            rgbs_folder=rgbs_folder,
            depths_folder=depths_folder,
            colls_folder=colls_folder,
            pretrained_vae_path=pretrained_vae_path,
            dc_beta = dc_beta,
            latent_dim=64,
            beta_coeff=0.0,
            alpha_align=1.0,  # 可以调整这个参数来平衡重构损失和对齐损失
            epochs=30,
            batch_size=256,
            lr=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            run_name="rgbc_vae",
            seed=42,
        )