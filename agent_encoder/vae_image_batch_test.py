import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from Project_dataset import DepthCollisionDataset, preprocess_image


# 定义VAE损失函数
def vae_loss_function2(recon_x, x, mean, logvar, beta=3):
    recon_loss = nn.functional.mse_loss(recon_x, x, size_average=False)
    kld_loss = -0.5 * torch.sum(1 + logvar.clamp(min=-10, max=10) - mean.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kld_loss
    return total_loss, recon_loss, kld_loss

# 针对图像缩放beta的损失函数
def vae_loss_function(recon_x, x, mean, logvar, beta_coeff=1.0, latent_dims=64):
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    MSE_LOSS = nn.MSELoss(reduction="none")
    cross_ent = MSE_LOSS(recon_x, x) * invalid_pixel_mask
    reconstruction_loss = torch.mean(torch.sum(cross_ent, dim=[1, 2, 3]))   # 11037
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))
    beta_norm = (beta_coeff*latent_dims)/(480.0*270.0)
    total_loss = reconstruction_loss + beta_norm * kld_loss
    return total_loss, reconstruction_loss, kld_loss


# 测试VAE模型
def test_vae(model_path, latent_dims, device="cuda", batch_size=256):
    # 加载预训练模型
    if model_path =="weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth":
        from ICRA_VAE import VAE
        model = VAE(input_dim=1, latent_dim=latent_dims, with_logits=False, inference_mode=True).to(device)
    else:
        from VAE import VAE
        model = VAE(input_dim=1, latent_dim=latent_dims, with_logits=False, inference_mode=True).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 准备测试数据集
    depths_folder = "/home/niu/workspaces/VAE_ws/datasets/depths"
    colls_folder = "/home/niu/workspaces/VAE_ws/datasets/colls_offset"
    # test_len = 1000     # 数据集最后1000张图片用于测试 todo

    # 准备数据集
    dataset = DepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)  # , test_len=test_len)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # 初始化损失函数
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0

    # 重构图像并计算损失
    with torch.no_grad():
        for depth_batch, coll_batch in test_dataloader:
            batch = depth_batch.to(device)
            coll_batch = coll_batch.to(device)
            recon_batch, mean, logvar, _ = model(batch)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, coll_batch, mean, logvar)
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_samples += batch.size(0)

    # 计算平均损失
    average_loss = total_loss / total_samples
    average_recon_loss = total_recon_loss / total_samples
    average_kl_loss = total_kl_loss / total_samples

    return average_loss, average_recon_loss, average_kl_loss, batch, coll_batch, recon_batch

# 可视化原始图像与重构图像
def plot_results(depth, original, reconstructed, n_images=5):
    depth = depth.cpu().numpy()
    original = original.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    plt.figure(figsize=(18, 6))
    for i in range(n_images):
        # 显示原始深度图像
        plt.subplot(3, n_images, i + 1)
        plt.imshow(depth[i, 0], cmap="gray")
        plt.title("Input Image")
        plt.axis("off")

        # 显示原始碰撞图像
        plt.subplot(3, n_images, i + 1+ n_images)
        plt.imshow(original[i, 0])
        plt.title("Collision")
        plt.axis("off")

        # 显示重构后的图像
        plt.subplot(3, n_images, i + 1 + 2*n_images)
        plt.imshow(reconstructed[i, 0])
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# 主程序
# 该程序用于测试 VAE 模型的（深度2膨胀）重构性能
if __name__ == "__main__":
    # 参数设置
    mark = 520    # 1=0.0078  2=0.0074  3=0.0089    5=0.0099    6=0.0098
    BETA = 10
    latent_dims = 64
    epoch_off = 10  # 修改后的 epoch_off 值

    # model_path = "weights/beta_vae_300.pth"   # vanilla VAE 没有碰撞编码
    model_path = f"weights/{mark}_beta_{BETA}_LD_{latent_dims}_epoch_{int(epoch_off)}.pth"  # 0.0078
    # model_path = "weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"  # 0.0132
    #ICRA数据拟合： 见 input_r_loss_results.txt 文件测试结果
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试模型
    average_loss, average_recon_loss, average_kl_loss, depth, collision, reconstructed = test_vae(model_path, latent_dims, device=device)

    # 打印平均重构损失和KL散度损失
    print(f"Average Reconstruction Loss: {average_recon_loss:.4f}")
    print(f"Average KL Loss: {average_kl_loss:.4f}")
    print(f"Average Total Loss: {average_loss:.4f}")

    # 可视化结果
    plot_results(depth, collision, reconstructed, n_images=5)