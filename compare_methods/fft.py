import cv2
from matplotlib import pyplot as plt
from scipy.fft import fft2, ifft2
import numpy as np


class FFT:
    def __init__(self):
        self.image = None
        self.fft = None
        self.reconstructed_image = None

    def get_image_reconstruction_with_compressed_dimensions(self, image, latent_dims=128):
        pixels = image.shape[0] * image.shape[1]
        self.fft = fft2(image)

        self.thresh_fft = self.fft.copy()
        self.fft_magnitude = np.abs(self.fft).flatten()
        self.fft_magnitude.sort()
        threshold = self.fft_magnitude[-latent_dims]
        self.thresh_fft[np.abs(self.fft) < threshold] = 0
        self.reconstructed_image = ifft2(self.thresh_fft).real  # Check if real is to be used or abs
        return self.fft, self.reconstructed_image

    def forward(self, image, latent_dims=128):
        return self.get_image_reconstruction_with_compressed_dimensions(image, latent_dims)[1]

def visualize(depth, re_depth):
    """
    可视化重构结果

    参数:
        result (dict): forward()的输出结果
        show_latent (bool): 是否显示隐特征统计信息
    """
    plt.figure(figsize=(12, 6))

    # 原始深度图
    plt.subplot(1, 2, 1)
    plt.imshow(depth, cmap="jet")
    plt.title("Original Depth")
    plt.axis("off")

    # 重构结果
    plt.subplot(1, 2, 2)
    plt.imshow(re_depth, cmap="gray")
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    depth_input = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36007.png"
    image = cv2.imread(depth_input, cv2.IMREAD_UNCHANGED).astype(np.float32)
    fft_processor = FFT()
    reconstructed_image = fft_processor.forward(image, latent_dims=128)

    print("Original Image Shape:", image.shape)
    print("Reconstructed Image Shape:", reconstructed_image.shape)
    print("Reconstruction Complete.")

    visualize(image, reconstructed_image)