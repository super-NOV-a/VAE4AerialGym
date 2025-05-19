import math
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard的SummaryWriter
from VAE import VAE
from tqdm import tqdm
import random
from Project_dataset import DepthCollisionDataset, preprocess_image  # 从单独的文件导入数据集类

# 设置随机数种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 对所有GPU设置相同的随机种子
torch.backends.cudnn.deterministic = True  # 确保CUDA使用确定性算法
torch.backends.cudnn.benchmark = False  # 禁用CuDNN自动寻找最合适的算法


def vae_loss_function(recon_x, x, mean, logvar, beta=3):
    """VAE损失函数"""
    recon_loss = nn.functional.mse_loss(recon_x, x, size_average=False)
    kld_loss = -0.5 * torch.sum(1 + logvar.clamp(min=-10, max=10) - mean.pow(2) - logvar.exp())
    kld_loss = beta * kld_loss
    return recon_loss + kld_loss, kld_loss


def MSE_KLD_Loss_unweighted_for_invalid_pixels(recon_x, x, mean, logvar, beta_coeff=3.0, latent_dims=64):
    """针对无效像素的MSE和KLD损失，无效像素值是0"""
    # invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(recon_x, x)    #  * invalid_pixel_mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_coeff = (beta_coeff*latent_dims)/(480.0*270.0)
    return reconstruction_loss + kld_loss * beta_coeff, kld_loss


def train_vae(model, dataloader, optimizer, epochs, device, lr_scheduler, beta=3, latent_dim=64, name=75, writer=None):
    """训练VAE模型"""
    model.train()
    loss_file = f"train_loss/{name}_beta_{beta}_LD_{latent_dim}.csv"
    with open(loss_file, 'w') as f:
        f.write("Epoch,Average Total Loss,Average KL Loss\n")

    for epoch in range(epochs):
        total_loss_sum = 0.0
        total_kl_loss = 0.0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for batch_idx, (depth_batch, coll_batch) in enumerate(dataloader):
                depth_batch = depth_batch.to(device)
                coll_batch = coll_batch.to(device)
                recon_batch, mean, logvar, _ = model(depth_batch)
                total_loss, kld_loss = MSE_KLD_Loss_unweighted_for_invalid_pixels(
                    recon_batch, coll_batch, mean, logvar, beta_coeff=beta, latent_dims=latent_dim)

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss_sum += total_loss.item()
                total_kl_loss += kld_loss.item()

                pbar.set_postfix(total_loss=total_loss.item(), kl_loss=kld_loss.item())
                pbar.update(1)

                # 使用TensorBoard记录损失
                if writer and batch_idx % 10 == 0:
                    writer.add_scalar('Loss/total_loss', total_loss.item(), epoch * len(dataloader) + batch_idx)
                    writer.add_scalar('Loss/kld_loss', kld_loss.item(), epoch * len(dataloader) + batch_idx)

        avg_total_loss = total_loss_sum / len(dataloader.dataset)
        avg_kl_loss = total_kl_loss / len(dataloader.dataset)

        print(
            f"Epoch {epoch + 1}/{epochs}, Average Total Loss: {avg_total_loss:.4f}, Average KL Loss: {avg_kl_loss:.4f}")
        with open(loss_file, 'a') as f:
            f.write(f"{epoch + 1},{avg_total_loss:.4f},{avg_kl_loss:.4f}\n")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"weights/{name}_beta_{beta}_LD_{latent_dim}_epoch_{epoch + 1}.pth")
            print(f"模型已保存至 weights/{name}_beta_{beta}_LD_{latent_dim}_epoch_{epoch + 1}.pth")

        lr_scheduler.step()

        # 使用TensorBoard记录每个epoch的平均损失
        if writer:
            writer.add_scalar('Loss/epoch_avg_total_loss', avg_total_loss, epoch)
            writer.add_scalar('Loss/epoch_avg_kld_loss', avg_kl_loss, epoch)

            with torch.no_grad():
                depth_samples = depth_batch[0].cpu().numpy()
                coll_samples = coll_batch[0].cpu().numpy()
                recon_samples = recon_batch[0].cpu().numpy()

                # 创建合并图像（优化布局）
                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
                fig.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.9, bottom=0.1)

                axes[0].imshow(depth_samples[0], cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Original Depth", fontsize=8)
                axes[0].axis("off")

                axes[1].imshow(coll_samples[0], cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Collision Map", fontsize=8)
                axes[1].axis("off")

                axes[2].imshow(recon_samples[0], cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Reconstructed", fontsize=8)
                axes[2].axis("off")

                plt.tight_layout(pad=0.5)

                # 转换为 Tensor
                fig.canvas.draw()
                merged_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                merged_image = merged_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)

                merged_image_tensor = torch.from_numpy(merged_image).permute(2, 0, 1).float() / 255.0
                writer.add_image("Merged/Depth_Coll_Recon", merged_image_tensor, epoch, dataformats="CHW")

    if writer:
        writer.close()

def test_vae(model, dataloader, device, writer=None, epoch=0):
    """测试VAE模型"""
    model.eval()
    with torch.no_grad():
        for depth_batch, coll_batch in dataloader:
            coll_batch = coll_batch.to(device)
            depth_batch = depth_batch.to(device)
            recon_batch, _, _, _ = model(depth_batch)
            return depth_batch, coll_batch, recon_batch



if __name__ == "__main__":
    # 参数配置
    # name = 101   # 30 真实数据  //   100 for 仿真50k+真实20k beta10   101 beta3  102 beta1
    # 111 beta3  112 beta1  新处理方法（新的核而不是旧的）
    latent_dims = 64
    # beta = 10    # 调整 beta * KL损失的权重
    batch_size = 256
    num_epochs = 40
    learning_rate = 0.0002
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集路径
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"
    # depths_folder = "/home/niu/下载/new_depth_images"

    # 准备数据集
    dataset = DepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化 VAE 模型
    vae = VAE(input_dim=1, latent_dim=latent_dims).to(device)

    name = 520
    betas = [1, 3, 10]  # beta为1 kL下降较低
    for beta in betas:

        # 定义优化器
        optimizer = optim.Adam(vae.parameters(), lr=learning_rate, betas=(0.9, 0.99))

        # 定义学习率调度器
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        # 初始化TensorBoard
        writer = SummaryWriter(log_dir='runs/vae_experiment_' + str(name)+"_beta_" + str(beta))

        # 训练模型
        train_vae(vae, dataloader, optimizer, epochs=num_epochs, device=device, lr_scheduler=lr_scheduler,
                  beta=beta, latent_dim=latent_dims, name=name, writer=writer)

        # # 测试模型
        # depth_batch, coll_batch, recon_batch = test_vae(vae, dataloader, device, writer=writer, epoch=num_epochs)
        #
        # # 可视化
        # depth_batch = depth_batch.cpu().numpy()
        # coll_batch = coll_batch.cpu().numpy()
        # recon_batch = recon_batch.cpu().numpy()
        #
        # fig, axes = plt.subplots(3, 5, figsize=(15, 6))
        # for i in range(5):
        #     axes[0, i].imshow(depth_batch[i, 0], cmap=plt.cm.jet, vmin=0, vmax=1)
        #     axes[0, i].set_title("Original Depth")
        #     axes[0, i].axis('off')
        #
        #     axes[1, i].imshow(coll_batch[i, 0], cmap=plt.cm.jet, vmin=0, vmax=1)
        #     axes[1, i].set_title("Original Depth Collision")
        #     axes[1, i].axis('off')
        #
        #     axes[2, i].imshow(recon_batch[i, 0], cmap=plt.cm.jet, vmin=0, vmax=1)
        #     axes[2, i].set_title("Reconstructed Depth Collision")
        #     axes[2, i].axis('off')
        # plt.tight_layout()
        # plt.show()