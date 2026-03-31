import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from agent_encoder.Net.VAE import VAE
from tqdm import tqdm
import random
import os
from Project_dataset_show import RGBDepthCollisionDataset, preprocess_image


def MSE_KLD_Loss_unweighted_for_invalid_pixels(
        recon_d: torch.Tensor,  # 重构的深度图
        d: torch.Tensor,  # 原始深度图 (目标)
        mean: torch.Tensor,
        logvar: torch.Tensor,
        beta_coeff: float = 3.0,
        latent_dims: int = 64
):
    """
    针对原始深度图的MSE和KLD损失。
    返回: total_loss, reconstruction_loss, kld_loss。
    注意：这里按固定输入分辨率 480x270 进行 beta 缩放。
    """
    # 启用 mask，与 dc_VAE 保持一致
    invalid_pixel_mask = torch.where(d > 0, torch.ones_like(d), torch.zeros_like(d))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(recon_d, d) * invalid_pixel_mask  # 启用 mask

    reconstruction_loss_unweighted = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))

    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

    beta_eff = (beta_coeff * latent_dims) / (480.0 * 270.0)
    total_loss = reconstruction_loss_unweighted + kld_loss * beta_eff
    return total_loss, reconstruction_loss_unweighted, kld_loss


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average KL per latent dimension across the batch (nats). Shape: [latent_dim]."""
    kld_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kld_elem.mean(dim=0)


def train_vae(
        depths_folder: str,
        colls_folder: str, # (关键修改) 添加 colls_folder
        latent_dim: int,
        beta: float,
        epochs: int,
        batch_size: int,
        lr: float,
        device: str,
        name: str,
        seed: int
):
    """训练VAE模型，使其日志和随机性与 train_dc_VAE.py 对齐"""

    # 严格的随机性设置 (来自 train_dc_VAE.py)
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

    # Dataloader 创建
    dataset = RGBDepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker
    )

    # 初始化 VAE 模型
    model = VAE(input_dim=1, latent_dim=latent_dim).to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # 定义学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 更新 TensorBoard writer 的 log_dir 名称以反映 vanilla VAE
    log_dir_name = f"runs/{name}_beta{beta}_LD_{latent_dims}"
    writer = SummaryWriter(log_dir=log_dir_name)
    print(f"TensorBoard 日志将保存到: {log_dir_name}")

    model.train()

    global_step = 0
    for epoch in range(1, epochs + 1):  # 循环从 1 到 epochs
        epoch_total_loss_sum = 0.0
        epoch_recon_loss_sum = 0.0
        epoch_kl_loss_sum = 0.0

        with tqdm(total=len(dataloader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for batch_idx, (depth_batch, coll_batch) in enumerate(dataloader):
                depth_batch = depth_batch.to(device)
                # coll_batch 在 vanilla vae 中不使用
                recon_batch, mean, logvar, _ = model(depth_batch)

                # 调用更新后的损失函数，目标是 depth_batch (D->D)
                total_loss, recon_loss, kld_loss = MSE_KLD_Loss_unweighted_for_invalid_pixels(
                    recon_batch, depth_batch, mean, logvar, beta_coeff=beta, latent_dims=latent_dim)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 累加 epoch 损失
                epoch_total_loss_sum += total_loss.item()
                epoch_recon_loss_sum += recon_loss.item()
                epoch_kl_loss_sum += kld_loss.item()

                # 更新 tqdm 显示
                beta_eff = (beta * latent_dim) / (480.0 * 270.0)
                pbar.set_postfix(total=f"{total_loss.item():.4f}", rec=f"{recon_loss.item():.4f}",
                                 kl=f"{kld_loss.item():.4f}", beta_eff=f"{beta_eff:.6f}")
                pbar.update(1)

                # 使用TensorBoard记录 *每一步* 的损失，并匹配 dc_VAE 的名称
                if writer:
                    writer.add_scalar('Loss/step_total_with_scaled_beta', total_loss.item(), global_step)
                    writer.add_scalar('Loss/step_recon_mse_unweighted', recon_loss.item(), global_step)
                    writer.add_scalar('Loss/step_kl_nats_unweighted', kld_loss.item(), global_step)
                    writer.add_scalar('Meta/beta_eff', beta_eff, global_step)
                    writer.add_scalar('Meta/lr', optimizer.param_groups[0]["lr"], global_step)
                    # KL per‑dim histogram
                    kld_dim = kl_per_dim(mean, logvar).detach().cpu().numpy()
                    writer.add_histogram("KL/per_dim", kld_dim, global_step)

                global_step += 1

        # 计算 epoch 平均损失
        num_batches = len(dataloader)
        avg_total_loss = epoch_total_loss_sum / num_batches
        avg_recon_loss = epoch_recon_loss_sum / num_batches
        avg_kl_loss = epoch_kl_loss_sum / num_batches

        # 更新打印信息
        print(
            f"Epoch {epoch}/{epochs} | avg_total(beta_eff)={avg_total_loss:.6f} | avg_rec_mse={avg_recon_loss:.6f} | avg_kl_nats={avg_kl_loss:.6f}")

        # 保存模型权重
        if epoch % 10 == 0:
            os.makedirs("weights", exist_ok=True)
            model_path = f"weights/{name}_beta{beta}_LD_{latent_dim}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存至 {model_path}")

        lr_scheduler.step()

        # 使用TensorBoard记录每个epoch的平均损失，匹配 dc_VAE 名称
        if writer:
            writer.add_scalar('Loss/epoch_avg_total_with_scaled_beta', avg_total_loss, epoch)
            writer.add_scalar('Loss/epoch_avg_recon_mse_unweighted', avg_recon_loss, epoch)
            writer.add_scalar('Loss/epoch_avg_kl_nats_unweighted', avg_kl_loss, epoch)

            # 更新图像面板逻辑
            with torch.no_grad():
                # 使用 depth_batch 作为原始和目标
                depth_samples = depth_batch[0].cpu().numpy()
                recon_samples = recon_batch[0].cpu().numpy()

                # 创建合并图像（1x3: Depth / Target=Depth / Recon=Depth）
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
                fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.9, bottom=0.1)

                im_opts = {"cmap": "gray", "vmin": 0, "vmax": 1}

                axes[0].imshow(depth_samples[0], **im_opts)
                axes[0].set_title("Input Depth", fontsize=8)
                axes[0].axis("off")

                axes[1].imshow(depth_samples[0], **im_opts)  # GT 现在是 depth_samples
                axes[1].set_title("GT Depth (Input)", fontsize=8)
                axes[1].axis("off")

                axes[2].imshow(recon_samples[0], **im_opts)
                axes[2].set_title("Recon Depth", fontsize=8)
                axes[2].axis("off")

                plt.tight_layout(pad=0.5)

                # 转换为 Tensor
                fig.canvas.draw()
                merged_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                merged_image = merged_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                merged_image_tensor = torch.from_numpy(merged_image).permute(2, 0, 1).float() / 255.0
                writer.add_image("Images/depth_gt_recon", merged_image_tensor, epoch, dataformats="CHW")

    if writer:
        writer.close()


def test_vae(model, dataloader, device, writer=None, epoch=0):
    """测试VAE模型"""
    model.eval()
    with torch.no_grad():
        for depth_batch, coll_batch in dataloader:
            depth_batch = depth_batch.to(device)
            recon_batch, _, _, _ = model(depth_batch)
            return depth_batch, depth_batch, recon_batch  # 返回 depth 作为 GT


if __name__ == "__main__":
    # 数据集路径
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_target"

    # betas = [3.0, 100.0]  #
    # betas =  [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    betas = [3000.0, 10000.0]  # 可调整 beta 范围
    # some_latent_dims = [32, 128, 256]
    some_latent_dims = [64]

    for latent_dims in some_latent_dims:
        for beta in betas:
            print(f"\n--- 开始训练: Latent Dim = {latent_dims}, Beta = {beta} ---")

            # 训练模型
            train_vae(
                depths_folder=depths_folder,
                colls_folder=colls_folder,
                latent_dim=latent_dims,
                beta=beta,
                epochs=30,
                batch_size=256,
                lr=2e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                name="beta_vae",
                seed=42
            )