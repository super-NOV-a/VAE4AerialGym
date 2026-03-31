import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from agent_encoder.utils import preprocess_image

class DepthVAEReconstructor:
    """
    深度图变分自编码器重构器

    参数:
        model_path (str): VAE模型权重文件路径
        latent_dim (int): 隐空间维度
        device (str, optional): 计算设备('cuda'或'cpu')
        image_size (tuple): 输入图像尺寸(宽, 高)
        min_depth (int): 最小有效深度值
        max_depth (int): 最大深度值
    """

    def __init__(self, model_path, latent_dim, device=None,
                 image_size=(480, 270), min_depth=15, max_depth=255, inference_mode=False):
        # 设备配置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 图像参数
        self.image_size = image_size
        self.min_depth = min_depth
        self.max_depth = max_depth

        # 加载模型
        if "ICRA" in model_path or "icra" in model_path:
            from agent_encoder.Net.ICRA_VAE import VAE  # optional path in your codebase
        else:
            from agent_encoder.Net.VAE import VAE
        self.model = VAE(input_dim=1, latent_dim=latent_dim, inference_mode=inference_mode).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def preprocess(self, depth_image):
        depth_image = preprocess_image(depth_image)
        # 转换为张量
        depth_tensor = depth_image.astype(np.float32) / 255.0
        return torch.tensor(depth_tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, depth_input, coll_input=None):
        """
        执行前向传播重构 - 修复coll_input处理
        """
        try:
            # 处理深度图输入
            if isinstance(depth_input, str):
                depth_image = cv2.imread(depth_input, cv2.IMREAD_UNCHANGED).astype(np.float32)
                # print(f"DEBUG: 从文件加载深度图: {depth_input}, 形状: {depth_image.shape}")
            else:
                depth_image = depth_input.astype(np.float32)
                # print(f"DEBUG: 使用数组深度图, 形状: {depth_image.shape}")

            # 处理碰撞图输入 - 更健壮的处理
            if coll_input is not None:
                if isinstance(coll_input, str):
                    coll_image = cv2.imread(coll_input, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    # print(f"DEBUG: 从文件加载碰撞图: {coll_input}, 形状: {coll_image.shape}")
                else:
                    coll_image = coll_input.astype(np.float32)
                    # print(f"DEBUG: 使用数组碰撞图, 形状: {coll_image.shape}")
            else:
                # 如果没有提供碰撞图，创建一个空的占位符
                coll_image = np.zeros_like(depth_image)
                # print(f"DEBUG: 创建空的碰撞图占位符, 形状: {coll_image.shape}")

            # 预处理
            # print(f"DEBUG: 开始预处理深度图")
            depth_tensor = self.preprocess(depth_image)
            # print(f"DEBUG: 预处理后张量形状: {depth_tensor.shape}")

            # 模型推理
            # print(f"DEBUG: 开始模型推理")
            with torch.no_grad():
                recon, mean, logvar, z = self.model(depth_tensor)
            # print(f"DEBUG: 模型推理完成, 重建图形状: {recon.shape}")

            # 转换为numpy
            recon_image = recon.squeeze().cpu().numpy()
            # print(f"DEBUG: 转换为numpy后形状: {recon_image.shape}")

            return depth_image, coll_image, recon_image

        except Exception as e:
            # print(f"ERROR: forward方法执行失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回错误状态
            error_image = np.zeros(self.image_size, dtype=np.float32)
            return error_image, error_image, error_image

def visualize(depth, coll, re_coll):
    """
    可视化重构结果

    参数:
        result (dict): forward()的输出结果
        show_latent (bool): 是否显示隐特征统计信息
    """
    plt.figure(figsize=(12, 6))

    # 原始深度图
    plt.subplot(1, 3, 1)
    plt.imshow(depth, cmap="jet")
    plt.title("Original Depth")
    plt.axis("off")

    # 原始深度图
    plt.subplot(1, 3, 2)
    plt.imshow(coll, cmap="jet")
    plt.title("Original Collision")
    plt.axis("off")

    # 重构结果
    plt.subplot(1, 3, 3)
    plt.imshow(re_coll, cmap="jet")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化重构器
    reconstructor = DepthVAEReconstructor(
        # model_path="/home/niu/workspaces/VAE_ws/agent_encoder/weights/990_beta_3_LD_64_epoch_40.pth",
        model_path="/home/niu/workspaces/VAE_ws/agent_encoder/weights/dc_vae_beta1.0_LD_64_epoch_30.pth",
        # model_path="/home/niu/workspaces/VAE_ws/agent_encoder/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth",
        latent_dim=64
    )

    # 执行重构
    depth, coll, re_depth = reconstructor.forward("/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png",
                                            "/home/niu/workspaces/VAE_ws/datasets/colls_target/depth_36007.png")
    # coll_input = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png"
    # depth_image = cv2.imread(coll_input, cv2.IMREAD_UNCHANGED).astype(np.float32)

    # 可视化结果
    visualize(depth, coll, re_depth)