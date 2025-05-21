import argparse
import os
import sys
import math
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import PineappleDataset
from diffusion_conv import Diffusion
from ddpm import DDPMSampler
import wandb
import numpy as np
#get the current working directory
current_dir = os.getcwd()
path_to_add = os.path.join(current_dir,"VAE_training")
# check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from VAE_training.VAE import VAE

def setup_wandb(lr,epochs,batch_size):
    """Login to Weights & Biases and initialize a new run."""
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)
    run = wandb.init(
        entity="imagine-laboratory-conare",
        project="SD_training_exp1",
        config={
            "learning_rate": lr,
            "architecture": "stable_diffusion",
            "dataset": "Pineapples",
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "Adam"
        },
    )
    return run
def get_time_embedding(timesteps: torch.LongTensor, dim: int = 160):
    """
    Create sinusoidal time embeddings for a batch of timesteps.
    Args:
        timesteps: (B,) LongTensor of timesteps
        dim: int, half dimension for sin+cos (output dim will be 2*dim)
    Returns:
        Tensor of shape (B, 2*dim)
    """
    device = timesteps.device
    half_dim = dim
    freqs = torch.pow(10000, -torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent-space DDPM on VAE latents")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Glob path to pineapple images, e.g. '/path/to/*' ")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--chkps_logging_path", type=str, default="checkpoints/diffusion/betaKL@1.0/",
                        help="Directory to save checkpoint files")
    parser.add_argument("--vae_chkp", type=str, default="checkpoints/vae/betaKL@1.0/weights_ck_398.pt",
                        help="Directory to save checkpoint files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=-1,
                        help="Early stopping patience in epochs")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping delta in epochs")
    parser.add_argument("--do_wandb", action="store_true",default=True,
                        help="Enable Weights & Biases logging")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.do_wandb:
        setup_wandb(args.lr,args.epochs,args.batch_size)
    os.makedirs(args.chkps_logging_path, exist_ok=True)

    # Dataset & DataLoader
    dataset = PineappleDataset(train=True, train_ratio=0.8,
                               dataset_path=args.dataset_path)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and freeze VAE
    vae = VAE().to(device)
    ckpt = torch.load(args.vae_chkp, map_location=device)
    vae.load_state_dict(ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Diffusion model
    diffusion_model = Diffusion().to(device)

    # DDPM sampler for noise schedule
    sampler = DDPMSampler(generator=torch.Generator(device=device),
                          num_training_steps=1000)

    # Optimizer
    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    # Cosine Annealing LR Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    mse_loss = torch.nn.MSELoss()

    T = sampler.num_train_timesteps
    global_step = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    es_min_delta = args.es_min_delta
    
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_loss_generation = 0.0
        epoch_loss_visual_latents = 0.0
        diffusion_model.train()
        epoch_losses = []
        with tqdm(total=len(loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
            for batch in loader:
                imgs = batch['image'].to(device)

                # 1. Encode to latent
                with torch.no_grad():
                    b, _, h, w = imgs.shape
                    noise_vae = torch.randn((b, 4, h // 8, w // 8), device=device)
                    latent, mu, logvar = vae.encoder(imgs, noise_vae)
                    latent = latent.detach()

                # 2. Sample per-sample timesteps and add noise
                t = torch.randint(0, T, (b,), device=device)
                noisy_lat,actual_noise,sqrt_alpha_prod,sqrt_one_minus_alpha_prod = sampler.add_noise(latent, t)

                # 3. Time embedding and prediction
                t_emb = get_time_embedding(t).to(device)           # (B, 320)
                pred_noise = diffusion_model(noisy_lat, t_emb)

                noisy_samples = sqrt_alpha_prod * latent + sqrt_one_minus_alpha_prod * pred_noise
                actual_noise_predicted = (noisy_samples - (sqrt_alpha_prod * latent)) / sqrt_one_minus_alpha_prod
                cleaned_latents_approximation = noisy_lat - actual_noise_predicted
                # 4. Loss & backward
                loss_generation = mse_loss(pred_noise, actual_noise)
                loss_visual_latents = mse_loss(cleaned_latents_approximation, latent)

                loss = loss_generation + loss_visual_latents
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  
                # 1) Log the batch loss
                global_step += 1
                if args.do_wandb:
                    wandb.log(
                        {
                            "train/batch_loss": loss.item(),
                            "train/batch_loss_generation": loss_generation.item(),
                            "train/batch_loss_visual_latents": loss_visual_latents.item(),
                            "train/learning_rate": optimizer.param_groups[0]['lr']
                        },
                        step=global_step
                    )
                loss_num = loss.item()
                epoch_losses.append(loss_num)
                epoch_loss += loss_num
                epoch_loss_generation += loss_generation.item()
                epoch_loss_visual_latents += loss_visual_latents.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        avg_loss = epoch_loss / len(loader)
        avg_loss_generation = epoch_loss_generation / len(loader)
        avg_loss_visual_latents = epoch_loss_visual_latents / len(loader)
        epoch_loss_std = np.std(epoch_losses)
        print(f"Epoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f} ± {epoch_loss_std:.4f}")
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_loss_generation": avg_loss_generation,
            "train/epoch_loss_visual_latents": avg_loss_visual_latents,
            "train/epoch_loss_std": epoch_loss_std,
            "train/epoch_lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        }, step=global_step)
        # Check for improvement
        if avg_loss + es_min_delta < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            chkpt_path = os.path.join(
                args.chkps_logging_path,
                f"diffusion_best_epoch{epoch}_loss{avg_loss:.4f}.pt"
            )
            torch.save(diffusion_model.state_dict(), chkpt_path)
            print(f"  ↳ New best model saved to {chkpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping
        if args.patience != -1:
            if epochs_no_improve >= args.patience:
                print(f"Stopping early after {epoch} epochs (patience {args.patience})")
                break
    # final save
    final_chkpt_path = os.path.join(
        args.chkps_logging_path,
        f"diffusion_final_epoch{epoch}_loss{avg_loss:.4f}.pt"
    )
    torch.save(diffusion_model.state_dict(), final_chkpt_path)
    print(f"Final model saved to {final_chkpt_path}")
    print("Training complete.")


if __name__ == '__main__':
    main()