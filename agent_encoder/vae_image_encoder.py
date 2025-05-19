import torch
import os
from VAE import VAE


def clean_state_dict(state_dict):
    # 该函数的作用是将state_dict中的key中的"module."和"dronet."替换为 空 和"encoder."
    clean_dict = {}
    for key, value in state_dict.items():
        if "module." in key:
            key = key.replace("module.", "")
        if "dronet." in key:
            key = key.replace("dronet.", "encoder.")
        clean_dict[key] = value
    return clean_dict


class VAEImageEncoder:
    """
    Class that wraps around the VAE class for efficient inference for the aerial_gym class
    这个类是为了在aerial_gym类中进行高效的推理而包装的VAE类
    导入了一个训练好的VAE模型，然后对输入的图像进行编码，返回编码后的图像
    """
    def __init__(self, config, device="cuda:0"):
        self.config = config
        self.vae_model = VAE(input_dim=1, latent_dim=self.config.latent_dims).to(device)
        # combine module path with model file name
        weight_file_path = os.path.join(self.config.model_folder, self.config.model_file)
        # load model weights
        print("Loading weights from file: ", weight_file_path)
        state_dict = clean_state_dict(torch.load(weight_file_path))
        self.vae_model.load_state_dict(state_dict)
        self.vae_model.eval()

    def encode(self, image_tensors):
        """
        Class to encode the set of images to a latent space. We can return both the means and sampled latent space variables.
        (batch_size, 270, 480, channels) -> (batch_size, latent_dims)
        深度图是 (batch_size 270 480)
        """
        with torch.no_grad():
            # need to squeeze 0th dimension and unsqueeze 1st dimension to make it work with the VAE
            image_tensors = image_tensors.squeeze(0).unsqueeze(1)
            x_res, y_res = image_tensors.shape[-2], image_tensors.shape[-1]
            if self.config.image_res != (x_res, y_res):
                # interpolate用于调整图像的大小，大小不符时 这里调整为270*480
                interpolated_image = torch.nn.functional.interpolate(
                    image_tensors,
                    self.config.image_res,
                    mode=self.config.interpolation_mode,
                )
            else:
                interpolated_image = image_tensors
            z_sampled, means, *_ = self.vae_model.encode(interpolated_image)
        if self.config.return_sampled_latent:
            returned_val = z_sampled
        else:
            returned_val = means
        return returned_val

    def decode(self, latent_spaces):
        """
        Decode a latent space to reconstruct full images
        """
        with torch.no_grad():
            if latent_spaces.shape[-1] != self.config.latent_dims:
                print(
                    f"ERROR: Latent space size of {latent_spaces.shape[-1]} does not match network size {self.config.latent_dims}"
                )
            decoded_image = self.vae_model.decode(latent_spaces)
        return decoded_image

    def get_latent_dims_size(self):
        """
        Function to get latent space dims
        """
        return self.config.latent_dims
