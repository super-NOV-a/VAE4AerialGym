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

from Project_dataset_show import RGBDepthCollisionDataset, preprocess_image
# === 1. 导入确定性 AE ===
from agent_encoder.Net.AE import AE


# ==============================================================
# 确定性 AE 的损失函数 (只有 MSE)
# ==============================================================

def MSE_Loss_for_invalid_pixels(
        recon_x: torch.Tensor,
        x: torch.Tensor
):
    """
    计算考虑无效像素 (x=0) 的 MSE 损失。
    """
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(recon_x, x) * invalid_pixel_mask  # 启用mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    return reconstruction_loss


# ==============================================================
# Training
# ==============================================================

def train(
        depths_folder: str,
        colls_folder: str,
        latent_dim: int = 64,
        # beta_coeff 已移除, AE中不需要
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 2e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        run_name: str = "dc_ae_deterministic",  # 修改 run_name
        seed: int = 42,
):
    # Reproducibility - 最严格的设置
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

    # Data - 同时使用生成器和worker初始化
    dataset = RGBDepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
        worker_init_fn=seed_worker
    )

    # === 2. 使用 AE 模型 ===
    model = AE(input_dim=1, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # TB
    os.makedirs("weights", exist_ok=True)
    # (beta_coeff 从日志名称中移除)
    writer = SummaryWriter(log_dir=f"runs/{run_name}_LD_{latent_dim}")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total, epoch_rec, epoch_kl = 0.0, 0.0, 0.0
        with tqdm(total=len(loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for depth_batch, coll_batch in loader:
                depth_batch = depth_batch.to(device)
                coll_batch = coll_batch.to(device)

                # === 3. AE forward pass ===
                # (AE 不返回 mu, logvar)
                recon, _z, _, _ = model(depth_batch)

                # === 4. AE 损失计算 ===
                rec_mse_unw = MSE_Loss_for_invalid_pixels(
                    recon_x=recon, x=coll_batch
                )

                # 为了与 VAE 训练日志保持一致, 我们显式地将 KLD 设为 0
                kld_loss = torch.tensor(0.0, device=device)

                # AE 的总损失就是重构损失
                total = rec_mse_unw

                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # 累加用于 epoch 平均
                epoch_total += total.item()
                epoch_rec += rec_mse_unw.item()
                epoch_kl += kld_loss.item()  # (将累加 0)

                # === 5. 日志记录 (保持名称不变) ===
                # KLD 和 beta 相关的日志将始终为 0
                beta_eff = 0.0  # AE 中 beta 始终为 0

                writer.add_scalar("Loss/step_recon_mse_unweighted", rec_mse_unw.item(), global_step)
                writer.add_scalar("Loss/step_kl_nats_unweighted", kld_loss.item(), global_step)  # (Log 0)
                writer.add_scalar("Loss/step_total_with_scaled_beta", total.item(), global_step)
                writer.add_scalar("Meta/beta_eff", beta_eff, global_step)  # (Log 0)
                writer.add_scalar("Meta/lr", optimizer.param_groups[0]["lr"], global_step)

                # (AE 没有 mu/logvar, 移除 KL per-dim 直方图)
                # writer.add_histogram("KL/per_dim", ...)

                pbar.set_postfix(total=f"{total.item():.4f}", rec=f"{rec_mse_unw.item():.4f}",
                                 kl=f"{kld_loss.item():.4f}", beta_eff=f"{beta_eff:.6f}")
                pbar.update(1)
                global_step += 1

        scheduler.step()

        # Epoch averages
        n_batches = len(loader)
        avg_total = epoch_total / n_batches
        avg_rec = epoch_rec / n_batches
        avg_kl = epoch_kl / n_batches  # (将为 0)
        print(
            f"Epoch {epoch}/{epochs} | avg_total(AE)={avg_total:.6f} | avg_rec_mse={avg_rec:.6f} | avg_kl_nats={avg_kl:.6f}")

        # === 6. Epoch 日志 (保持名称不变) ===
        writer.add_scalar("Loss/epoch_avg_total_with_scaled_beta", avg_total, epoch)
        writer.add_scalar("Loss/epoch_avg_recon_mse_unweighted", avg_rec, epoch)
        writer.add_scalar("Loss/epoch_avg_kl_nats_unweighted", avg_kl, epoch)  # (Log 0)

        # Epoch image panel (保持不变)
        with torch.no_grad():
            d0 = depth_batch[:1].cpu().numpy()[0, 0]
            g0 = coll_batch[:1].cpu().numpy()[0, 0]
            r0 = recon[:1].cpu().numpy()[0, 0]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            axes[0].imshow(d0, cmap="gray", vmin=0, vmax=1);
            axes[0].set_title("Depth");
            axes[0].axis("off")
            axes[1].imshow(g0, cmap="gray", vmin=0, vmax=1);
            axes[1].set_title("GT collision");
            axes[1].axis("off")
            axes[2].imshow(r0, cmap="gray", vmin=0, vmax=1);
            axes[2].set_title("Recon");
            axes[2].axis("off")
            plt.tight_layout()
            fig.canvas.draw()
            merged = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            merged = merged.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            merged_t = torch.from_numpy(merged).permute(2, 0, 1).float() / 255.0
            writer.add_image("Images/depth_gt_recon", merged_t, epoch, dataformats="CHW")

        # Save ckpt
        if epoch % 10 == 0:
            ckpt = f"weights/dc_ae_LD_{latent_dim}_epoch_{epoch}.pth"  # (移除 beta)
            torch.save(model.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    writer.close()


if __name__ == "__main__":
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_target"

    # === 7. 移除 beta 循环 ===
    latent_dims = [32, 64, 128, 256]  # 可以测试不同的 latent_dims

    for latent in latent_dims:
        train(
            depths_folder=depths_folder,
            colls_folder=colls_folder,
            latent_dim=latent,
            epochs=30,
            batch_size=256,
            lr=2e-4,
            device="cuda" if torch.cuda.is_available() else "cpu",
            run_name="dc_ae_deterministic",  # 确保 run_name 匹配
            seed=42,
        )