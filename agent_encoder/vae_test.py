import os
import re
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Project_dataset_show import DepthCollisionDataset, preprocess_image


# ===== Loss identical to training (masked MSE + scaled-beta KL) =====
# Mirrors MSE_KLD_Loss_unweighted_for_invalid_pixels in train_dc_VAE.py
# reconstruction_loss = mean( sum_{C,H,W} (recon-x)^2 * mask(x>0) ) over batch
# kld_loss           = mean( sum_{latent} KL ) over batch (in nats)
# total              = reconstruction_loss + ((beta*LD)/(480*270)) * kld_loss

def mse_kld_loss_masked_scaled_beta(recon_x, x, mean, logvar, beta_coeff=3.0, latent_dims=64):
    invalid_pixel_mask = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    mse_map = F.mse_loss(recon_x, x, reduction="none") * invalid_pixel_mask
    reconstruction_loss = torch.sum(mse_map, dim=(1, 2, 3)).mean()
    kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
    beta_eff = (beta_coeff * float(latent_dims)) / (480.0 * 270.0)
    total = reconstruction_loss + beta_eff * kld_loss
    return total, reconstruction_loss, kld_loss, beta_eff


# ===== Utility: parse beta & latent_dim from model filename =====
BETA_PATTERNS = [
    r"kld_beta[_-]?([0-9]+(?:\.[0-9]+)?)",
    r"beta[_-]?([0-9]+(?:\.[0-9]+)?)",
    r"BETA[_-]?([0-9]+(?:\.[0-9]+)?)",
]
LD_PATTERNS = [
    r"LD[_-]?([0-9]+)",
    r"latent[_-]?(?:dim|dims)?[_-]?([0-9]+)",
]

def parse_config_from_name(path: str, default_beta: float = 3.0, default_ld: int = 64):
    name = os.path.basename(path)
    beta = None
    ld = None
    for pat in BETA_PATTERNS:
        m = re.search(pat, name)
        if m:
            try:
                beta = float(m.group(1))
                break
            except Exception:
                pass
    for pat in LD_PATTERNS:
        m = re.search(pat, name)
        if m:
            try:
                ld = int(m.group(1))
                break
            except Exception:
                pass
    return (beta if beta is not None else default_beta), (ld if ld is not None else default_ld)


# ===== Model loader =====

def load_model(model_path: str, latent_dim: int, device: torch.device):
    if "ICRA" in model_path or "icra" in model_path:
        from agent_encoder.Net.ICRA_VAE import VAE  # optional path in your codebase
    else:
        from agent_encoder.Net.VAE import VAE
    model = VAE(input_dim=1, latent_dim=latent_dim).to(device)
    sd = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception:
        # allow loading checkpoints that wrapped extra keys
        if isinstance(sd, dict) and "state_dict" in sd:
            model.load_state_dict(sd["state_dict"], strict=False)
        else:
            model.load_state_dict(sd, strict=False)
    model.eval()
    return model


# ===== Visualization =====

def save_panel(depth, target, recon, save_path: str, n_cols: int = 5):
    """Save a panel: rows = [Depth, GT, Recon]. depth/target/recon are torch tensors [B,1,H,W] in [0,1]."""
    depth = depth.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    recon = recon.detach().cpu().numpy()

    n = min(n_cols, depth.shape[0])
    plt.figure(figsize=(3*n, 9))
    for i in range(n):
        # depth
        ax = plt.subplot(3, n, i + 1)
        ax.imshow(depth[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Depth")
        ax.axis("off")
        # target
        ax = plt.subplot(3, n, i + 1 + n)
        ax.imshow(target[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("GT")
        ax.axis("off")
        # recon
        ax = plt.subplot(3, n, i + 1 + 2*n)
        ax.imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title("Recon")
        ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()


# ===== Main evaluation =====

def evaluate(model_path: str,
             depths_folder: str = "/home/niu/workspaces/VAE_ws/data_test/depths",
             colls_folder: str = "/home/niu/workspaces/VAE_ws/data_test/colls_target",
             batch_size: int = 256,
             device_str: str = None,
             n_vis: int = 5,
             save_dir: str = "eval_dc_vae"):
    # parse beta & latent_dim from filename
    beta, latent_dim = parse_config_from_name(model_path)
    device = torch.device(device_str if device_str is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    # data
    dataset = DepthCollisionDataset(depths_folder, colls_folder, transform=preprocess_image)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model
    model = load_model(model_path, latent_dim, device)

    # accumulators (weighted by batch size)
    sum_rec = 0.0
    sum_kl = 0.0
    sum_total = 0.0
    n_samples = 0

    first_depth = None
    first_target = None
    first_recon = None

    with torch.no_grad():
        for depth_batch, coll_batch in loader:
            depth_batch = depth_batch.to(device)
            coll_batch = coll_batch.to(device)

            recon, mu, logvar, _ = model(depth_batch)
            total, rec, kl, beta_eff = mse_kld_loss_masked_scaled_beta(
                recon_x=recon, x=coll_batch, mean=mu, logvar=logvar,
                beta_coeff=beta, latent_dims=latent_dim,
            )

            B = depth_batch.size(0)
            sum_rec += rec.item() * B
            sum_kl  += kl.item() * B
            sum_total += total.item() * B
            n_samples += B

            if first_depth is None:
                first_depth = depth_batch[:n_vis]
                first_target = coll_batch[:n_vis]
                first_recon = recon[:n_vis]

    avg_rec = sum_rec / max(1, n_samples)
    avg_kl  = sum_kl  / max(1, n_samples)
    avg_total = sum_total / max(1, n_samples)

    print("==== dc‑VAE Test Results ====")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Parsed beta={beta}, latent_dim={latent_dim}, beta_eff={(beta*latent_dim)/(480.0*270.0):.6f}")
    print(f"Avg Reconstruction Loss (sum MSE per image): {avg_rec:.6f}")
    print(f"Avg KL Loss (nats per image):               {avg_kl:.6f}")
    print(f"Avg Total Loss (with scaled beta):           {avg_total:.6f}")

    # save a visualization panel
    panel_path = os.path.join(save_dir, "panel_depth_gt_recon.png")
    save_panel(first_depth, first_target, first_recon, panel_path, n_cols=n_vis)
    print(f"Saved visualization to: {panel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model .pth")
    parser.add_argument("--depths", type=str, default="/home/niu/workspaces/VAE_ws/data_test/depths")
    parser.add_argument("--colls", type=str, default="/home/niu/workspaces/VAE_ws/data_test/colls_target")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="eval_dc_vae")
    args = parser.parse_args()

    evaluate(model_path=args.model,
             depths_folder=args.depths,
             colls_folder=args.colls,
             batch_size=args.batch,
             device_str=args.device,
             n_vis=args.n_vis,
             save_dir=args.save_dir)
