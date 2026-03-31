import numpy as np
import pywt
from matplotlib import pyplot as plt


class WaveletTransforms:
    def __init__(self, name='db1', level=7):
        self.image = None
        self.coeffs = None
        self.reconstructed_image = None
        self.name = name
        self.level = level

    def get_wavelet_coefficients(self, image, latent_dims=128):
        self.image = image
        self.coeffs = pywt.wavedec2(image, wavelet=self.name, level=self.level)
        # select top latent_dims coefficients after sorting
        coeff_array, coeff_slices = pywt.coeffs_to_array(self.coeffs)
        Csort = np.sort(np.abs(coeff_array.reshape(-1)))
        threshold = Csort[-latent_dims]
        coeff_array[np.abs(coeff_array) < threshold] = 0
        coeffs_filt_array = pywt.array_to_coeffs(coeff_array, coeff_slices, output_format='wavedec2')
        self.reconstructed_image = pywt.waverec2(coeffs_filt_array, wavelet=self.name)
        return coeff_array, self.reconstructed_image

    def forward_with_latent_dims(self, image, latent_dims=128):
        return self.get_wavelet_coefficients(image, latent_dims)

    def get_reconstruction_with_latent_dims(self, latent_space, slices):
        coeffs_filt_array = pywt.array_to_coeffs(latent_space, slices, output_format='wavedec2')
        reconstructed_image = pywt.waverec2(coeffs_filt_array, wavelet=self.name)
        return reconstructed_image

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
    depth_input = "/home/niu/workspaces/VAE_ws/datasets/depths/depth_36008.png"
    image = plt.imread(depth_input).astype(np.float32)
    wavelet_processor = WaveletTransforms(name='db1', level=100)
    coeffs, reconstructed_image = wavelet_processor.forward_with_latent_dims(image, latent_dims=128)

    print("Original Image Shape:", image.shape)
    print("Reconstructed Image Shape:", reconstructed_image.shape)

    # 求一下重构图像的RMSE
    rmse = np.sqrt(np.mean((image - reconstructed_image) ** 2))
    print("RMSE of Reconstruction:", rmse)

    visualize(image, reconstructed_image)

