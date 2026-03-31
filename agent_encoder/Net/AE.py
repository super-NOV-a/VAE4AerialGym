import torch
import torch.nn as nn
from agent_encoder.Net.VAE import ImgDecoder  # 重用 VAE 的解码器


class ImgEncoderAE(nn.Module):
    """
    确定性 AE 的编码器 (Deterministic AE Encoder).
    与 VAE.py 中的 ImgEncoder 几乎相同,
    但最后一层只输出 'latent_dim' 而不是 '2 * latent_dim'.
    """

    def __init__(self, input_dim, latent_dim):
        """
        Parameters:
        ----------
        input_dim: int
            Number of input channels in the image.
        latent_dim: int
            Number of latent dimensions
        """
        super(ImgEncoderAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.define_encoder()
        self.elu = nn.ELU()
        print("Defined Deterministic AE Encoder.")

    def define_encoder(self):
        # define conv functions
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

        # === 唯一的修改 ===
        # VAE 输出 2*latent_dim (mu + logvar)
        # AE 只输出 latent_dim (z)
        self.dense1 = nn.Linear(512, self.latent_dim)
        # =================

        print("Encoder network initialized.")

    def forward(self, img):
        return self.encode(img)

    def encode(self, img):
        """
        Encodes the input image.
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

        # 直接返回 z
        return x


class AE(nn.Module):
    """确定性自编码器 (Deterministic Autoencoder)"""

    def __init__(self, input_dim=1, latent_dim=64, inference_mode=False):
        super(AE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.inference_mode = inference_mode  # (这个参数在AE中无用, 但保留以兼容)

        # 1. 使用确定性编码器
        self.encoder = ImgEncoderAE(input_dim=self.input_dim, latent_dim=self.latent_dim)

        # 2. 重用 VAE 的解码器
        self.img_decoder = ImgDecoder(
            input_dim=1, latent_dim=self.latent_dim, with_logits=False
        )

    def forward(self, img):
        """
        AE 的前向传播
        """
        # 1. 编码器直接输出 z
        z = self.encoder(img)

        # 2. 解码器重构
        img_recon = self.img_decoder(z)

        # 3. 返回 (recon, z) 以便训练脚本记录
        #    (我们返回 None 来填充 VAE 的 mu 和 logvar, 尽管我们不会使用它们)
        return img_recon, z, None, None

    def encode(self, img):
        z = self.encoder(img)
        # 返回 z, z, None 以匹配 VAE encode 的签名 (z_sampled, means, std)
        return z, z, None

    def decode(self, z):
        img_recon = self.img_decoder(z)
        return img_recon

    def set_inference_mode(self, mode):
        self.inference_mode = mode