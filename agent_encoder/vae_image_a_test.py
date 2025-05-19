import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from VAE import VAE  # 假设VAE模型在VAE.py中定义

# 设置设备（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (480, 270)
CAMERA_HFOV_DEG = 87
DRONE_SIZE_METERS = 0.5  # 无人机直径0.5米
# 无人机半径
DRONE_HALF_SIZE_METERS = DRONE_SIZE_METERS / 2
MAX_DILATION_SIZE = 50
MIN_DEPTH = int(0.6 * 255 / 10)  # 0.2对应 5  0.6对应 15
MAX_DEPTH = 255


def custom_normalize(normalized_depth_map):
    """
    将深度图归一化到[0, 255]范围，大于MAX_DEPTH设置为255 小于MIN_DEPTH的像素值设置为0
    返回值范围[0, 255]的深度图
    """
    max_val = np.max(normalized_depth_map)
    if max_val > 255:
        normalized_depth_map = normalized_depth_map * 255 / 7000  # DIML/CVl RGB-D 除以1000得到米  不再使用 SUN RGB-D
    # else:
    #     normalized_depth_map = normalized_depth_map
    normalized_depth_map[normalized_depth_map > MAX_DEPTH] = MAX_DEPTH
    normalized_depth_map[normalized_depth_map < MIN_DEPTH] = 0
    normalized_depth_map = cv2.resize(normalized_depth_map, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    return normalized_depth_map.astype(np.uint8)


model_path = "/home/niu/workspaces/VAE_ws/agent_encoder/weights/"
# model = "917_beta_10_LD_64_epoch_40.pth"
# model = "520_beta_10_LD_64_epoch_10.pth"
model = "ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"

if model == "ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth":
    from ICRA_VAE import VAE
# 加载VAE模型
vae = VAE(input_dim=1, latent_dim=64).to(device)
vae.load_state_dict(torch.load(model_path+model))
vae.eval()

# 加载深度图像
# depth_path = "/home/niu/workspaces/aerial_gym_ws/src/aerial_gym_simulator/aerial_gym/rl_training/rl_games/anomaly_images/anomaly_53.png"
depth_path = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_15339.png"
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

def preprocess_image(image):
    """将图像归一化到 [0.0, 1.0] 并转换为 PyTorch 张量"""
    image = image / 255.0
    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)
depth_image = custom_normalize(depth_image)
depth_tensor = preprocess_image(depth_image).unsqueeze(0).to(device)  # 添加批次维度

# 使用VAE进行编码和解码
with torch.no_grad():
    img_recon, mean, logvar, z_sampled = vae(depth_tensor)

# 将解码图像转换为NumPy数组
decoded_array = img_recon.squeeze().cpu().numpy()

# 绘制原图和解码图像
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(depth_image, cmap='gray')
ax[0].set_title('Original Depth Image')
ax[0].axis('off')

ax[1].imshow(decoded_array, cmap='gray', vmin=0, vmax=1)
ax[1].set_title('Decoded Depth Image')
ax[1].axis('off')
plt.tight_layout()
plt.show()

# 输出潜在特征的统计信息
print("Model:", model)
print("Latent Features Min:", torch.min(z_sampled))
print("Latent Features Max:", torch.max(z_sampled))
print("Latent Features Mean:", torch.mean(z_sampled))
print("Latent Features Std:", torch.std(z_sampled))