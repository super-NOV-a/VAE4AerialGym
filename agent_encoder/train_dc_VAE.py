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
from agent_encoder.Net.VAE import VAE  # expects forward(depth)-> (recon in [0,1], mu, logvar, z)


# ==============================================================
# EXACT loss as you specified (beta scaled by latent_dims/(480*270))
# ==============================================================

def MSE_KLD_Loss_unweighted_for_invalid_pixels(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    beta_coeff: float = 3.0,
    latent_dims: int = 64,
):
    """针对无效像素的MSE和KLD损失（按你提供的实现，不应用mask）。
    返回: total_loss, kld_loss。（重构项在训练外部单独统计到TB）
    注意：这里按固定输入分辨率 480x270 进行 beta 缩放。
    """
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(recon_x, x) * invalid_pixel_mask # 启用mask)
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_coeff = (beta_coeff * latent_dims) / (480.0 * 270.0)
    return reconstruction_loss + kld_loss * beta_coeff, reconstruction_loss, kld_loss


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average KL per latent dimension across the batch (nats). Shape: [latent_dim]."""
    kld_elem = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    return kld_elem.mean(dim=0)


# ==============================================================
# Training
# ==============================================================

def train(
    depths_folder: str,
    colls_folder: str,
    latent_dim: int = 64,
    beta_coeff: float = 3.0,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 2e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    run_name: str = "dc_vae_betaScaled_exact",
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
    # # Data
    # dataset = RGBDepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Model/optim
    model = VAE(input_dim=1, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # TB
    os.makedirs("weights", exist_ok=True)
    writer = SummaryWriter(log_dir=f"runs/{run_name}_beta{beta_coeff}_LD_{latent_dim}")

    def _recon_mse_unweighted(recon_x: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mse_map = F.mse_loss(recon_x, x, reduction="none")
        return torch.sum(mse_map, dim=(1, 2, 3)).mean()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total, epoch_rec, epoch_kl = 0.0, 0.0, 0.0
        with tqdm(total=len(loader), desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for depth_batch, coll_batch in loader:
                depth_batch = depth_batch.to(device)
                coll_batch  = coll_batch.to(device)

                recon, mu, logvar, _ = model(depth_batch)

                # === exact loss call ===
                total, rec_mse_unw, kld_loss = MSE_KLD_Loss_unweighted_for_invalid_pixels(
                    recon_x=recon, x=coll_batch, mean=mu, logvar=logvar,
                    beta_coeff=beta_coeff, latent_dims=latent_dim,
                )

                optimizer.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # For TB (beta-independent components)
                # rec_mse_unw = _recon_mse_unweighted(recon, coll_batch)

                epoch_total += total.item()
                epoch_rec   += rec_mse_unw.item()
                epoch_kl    += kld_loss.item()

                # Compute effective beta for logging (the same formula used inside loss)
                beta_eff = (beta_coeff * latent_dim) / (480.0 * 270.0)

                writer.add_scalar("Loss/step_recon_mse_unweighted", rec_mse_unw.item(), global_step)
                writer.add_scalar("Loss/step_kl_nats_unweighted",   kld_loss.item(),    global_step)
                writer.add_scalar("Loss/step_total_with_scaled_beta", total.item(),     global_step)
                writer.add_scalar("Meta/beta_eff", beta_eff, global_step)
                writer.add_scalar("Meta/lr", optimizer.param_groups[0]["lr"], global_step)

                # KL per‑dim histogram
                kld_dim = kl_per_dim(mu, logvar).detach().cpu().numpy()
                writer.add_histogram("KL/per_dim", kld_dim, global_step)

                pbar.set_postfix(total=f"{total.item():.4f}", rec=f"{rec_mse_unw.item():.4f}", kl=f"{kld_loss.item():.4f}", beta_eff=f"{beta_eff:.6f}")
                pbar.update(1)
                global_step += 1

        scheduler.step()

        # Epoch averages (beta‑independent for rec/kl)
        n_batches = len(loader)
        avg_total = epoch_total / n_batches
        avg_rec   = epoch_rec   / n_batches
        avg_kl    = epoch_kl    / n_batches
        print(f"Epoch {epoch}/{epochs} | avg_total(beta_eff)={avg_total:.6f} | avg_rec_mse={avg_rec:.6f} | avg_kl_nats={avg_kl:.6f}")

        writer.add_scalar("Loss/epoch_avg_total_with_scaled_beta", avg_total, epoch)
        writer.add_scalar("Loss/epoch_avg_recon_mse_unweighted",  avg_rec,   epoch)
        writer.add_scalar("Loss/epoch_avg_kl_nats_unweighted",    avg_kl,    epoch)

        # Epoch image panel: Depth / GT / Recon
        with torch.no_grad():
            d0 = depth_batch[:1].cpu().numpy()[0,0]
            g0 = coll_batch[:1].cpu().numpy()[0,0]
            r0 = recon[:1].cpu().numpy()[0,0]

            fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
            axes[0].imshow(d0, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Depth"); axes[0].axis("off")
            axes[1].imshow(g0, cmap="gray", vmin=0, vmax=1); axes[1].set_title("GT collision"); axes[1].axis("off")
            axes[2].imshow(r0, cmap="gray", vmin=0, vmax=1); axes[2].set_title("Recon"); axes[2].axis("off")
            plt.tight_layout()
            fig.canvas.draw()
            merged = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            merged = merged.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            merged_t = torch.from_numpy(merged).permute(2,0,1).float()/255.0
            writer.add_image("Images/depth_gt_recon", merged_t, epoch, dataformats="CHW")

        # Save ckpt
        if epoch % 10 == 0:
            ckpt = f"weights/dc_vae_beta{beta_coeff}_LD_{latent_dim}_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved: {ckpt}")

    writer.close()


if __name__ == "__main__":
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder  = "/home/niu/workspaces/VAE_ws/datasets/colls_target"

    # betas = [0.1, 0.3, 1.0, 10.0, 30.0, 300.0, 1000.0]  # 可调整 beta 范围
    betas = [3000.0, 10000.0]  # 可调整 beta 范围
    latent_dims = [64]    # 32, 128, 256
    for latent in latent_dims:
        for beta in betas:
            # 可按照需要改 beta_coeff / latent_dim
            train(
                depths_folder=depths_folder,
                colls_folder=colls_folder,
                latent_dim=latent,  # 64,
                beta_coeff=beta,    # 默认为100.0,
                epochs=30,
                batch_size=256,
                lr=2e-4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                run_name="dc_vae",
                seed=42,
            )
