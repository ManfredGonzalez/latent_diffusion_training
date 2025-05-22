import argparse
import os
import sys
import math
from collections import defaultdict

from tqdm import tqdm
import torchvision
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from Dataset import PineappleDataset
from diffusion_conv import Diffusion
from ddpm import DDPMSampler
import wandb
import numpy as np

# get the current working directory
current_dir = os.getcwd()
path_to_add = os.path.join(current_dir, "VAE_training")
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from VAE_training.VAE import VAE


def setup_wandb(lr, epochs, batch_size):
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
            "optimizer": "AdamW"
        },
    )
    return run


def get_time_embedding(timesteps: torch.LongTensor, dim: int = 160):
    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0)
    device = timesteps.device
    half_dim = dim
    freqs = torch.pow(
        10000,
        -torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb


def parse_args():
    parser = argparse.ArgumentParser(description="Train latent-space DDPM on VAE latents")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Glob path to pineapple images, e.g. '/path/to/*'")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--chkps_logging_path", type=str,
                        default="checkpoints/diffusion/betaKL@1.0/",
                        help="Directory to save checkpoint files")
    parser.add_argument("--vae_chkp", type=str,
                        default="checkpoints/vae/betaKL@1.0/weights_ck_398.pt",
                        help="Path to VAE checkpoint")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=-1, help="Early stopping patience")
    parser.add_argument("--rec_importance", type=int, default=1,
                        help="Reconstruction importance in loss")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping delta")
    parser.add_argument("--do_wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    return parser.parse_args()


'''def log_bar(name, loss_dict, step):
    """Helper to build a W&B bar plot from timestep→[losses]."""
    table = wandb.Table(columns=["timestep", "loss"])
    for t, vals in sorted(loss_dict.items()):
        table.add_data(t, float(np.mean(vals)))
    wandb.log({
        name: wandb.plot.bar(
            table, "timestep", "loss", title=name.replace("_", " ").title()
        )
    }, step=step)'''
def log_bar(name, loss_dict, step):
    """Helper to build a W&B bar plot from timestep→[losses]."""
    epoch = step
    table = wandb.Table(columns=["epoch", "timestep", "loss"])
    for t, vals in sorted(loss_dict.items()):
        table.add_data(epoch, t, float(np.mean(vals)))
    wandb.log({
        name: wandb.plot.bar(
            table, "timestep", "loss", title=name.replace("_", " ").title()
        )
    }, step=epoch)

def sample_i(h, w, vae, diffusion_model, generator, epoch, global_step, device, seed):
    generator.manual_seed(seed)
    sampler_i = DDPMSampler(generator)
    sampler_i.set_inference_timesteps(1000)
    with torch.no_grad():
        latents = torch.randn((1, 4, h // 8, w // 8), device=device)
        for timestep in tqdm(sampler_i.timesteps, desc="Sampling"):
            t = torch.tensor([int(timestep)], dtype=torch.long, device=device)
            time_embedding = get_time_embedding(t).to(device)
            model_output = diffusion_model(latents, time_embedding)
            latents = sampler_i.step(timestep, latents, model_output)

        # Decode & log
        decoded = vae.decoder(latents)
        img = decoded.squeeze(0).cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        '''# convert to float in [0,1]
        tensor = torch.from_numpy(img).permute(2,0,1).float().div(255.0)
        torchvision.utils.save_image(tensor, img_path)'''
        image = wandb.Image(img, caption=f"sample_epoch_{epoch}")


        wandb.log({"examples": image})

        


def main():
    args = parse_args()
    if args.do_wandb:
        setup_wandb(args.lr, args.epochs, args.batch_size)
    os.makedirs(args.chkps_logging_path, exist_ok=True)

    dataset = PineappleDataset(train=True, train_ratio=0.8, dataset_path=args.dataset_path)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load & freeze VAE
    vae = VAE().to(device)
    ckpt = torch.load(args.vae_chkp, map_location=device)
    vae.load_state_dict(ckpt)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    diffusion_model = Diffusion().to(device)
    sampler = DDPMSampler(generator=torch.Generator(device=device),
                          num_training_steps=1000)

    optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    T = sampler.num_train_timesteps
    # initialize uniform weights
    weights = torch.ones(T, dtype=torch.float32, device=device) / T
    global_step = 0
    best_loss = float('inf')
    epochs_no_improve = 0
    es_min_delta = args.es_min_delta

    for epoch in range(1, args.epochs + 1):
        diffusion_model.train()
        epoch_losses = []
        epoch_losses_gen = []
        epoch_losses_vis = []

        # per-timestep accumulators
        loss_gen_by_t = defaultdict(list)
        loss_vis_by_t = defaultdict(list)
        loss_tot_by_t = defaultdict(list)

        with tqdm(total=len(loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
            for batch in loader:
                imgs = batch['image'].to(device)

                # 1. Encode
                with torch.no_grad():
                    b, _, h, w = imgs.shape
                    noise_vae = torch.randn((b, 4, h // 8, w // 8), device=device)
                    latent, mu, logvar = vae.encoder(imgs, noise_vae)
                    latent = latent.detach()

                # 2. Sample t & add noise
                #t = torch.randint(0, T, (b,), device=device)
                #    draw b indices from [0..T-1] with prob ∝ w
                t = torch.multinomial(weights, num_samples=b, replacement=True).to(device)

                noisy_lat, actual_noise, sqrt_alpha_prod, sqrt_one_minus_alpha_prod = sampler.add_noise(latent, t)

                # 3. Time embed & predict
                t_emb = get_time_embedding(t).to(device)
                pred_noise = diffusion_model(noisy_lat, t_emb)

                # reconstruct & compute per-sample losses
                noisy_samples = sqrt_alpha_prod * latent + sqrt_one_minus_alpha_prod * pred_noise
                actual_noise_pred = (noisy_samples - sqrt_alpha_prod * latent) / sqrt_one_minus_alpha_prod

                # per-sample MSE
                per_gen = F.mse_loss(pred_noise, actual_noise, reduction="none").mean(dim=[1,2,3])
                per_vis = F.mse_loss(noisy_lat - actual_noise_pred, latent, reduction="none").mean(dim=[1,2,3])
                per_tot = per_gen + args.rec_importance * per_vis

                # batch-level loss
                loss_generation = per_gen.mean()
                loss_visual    = per_vis.mean()
                loss           = per_tot.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # accumulate for bar plots
                for ti, lg, lv, lt in zip(t.tolist(), per_gen.tolist(),
                                          per_vis.tolist(), per_tot.tolist()):
                    loss_gen_by_t[ti].append(lg)
                    loss_vis_by_t[ti].append(lv)
                    loss_tot_by_t[ti].append(lt)

                # log batch metrics
                global_step += 1
                '''if args.do_wandb:
                    wandb.log({
                        "train/batch_loss_total":   loss.item(),
                        "train/batch_loss_gen":     loss_generation.item(),
                        "train/batch_loss_visual":  loss_visual.item(),
                        "train/lr":                 optimizer.param_groups[0]['lr']
                    }, step=global_step)'''

                epoch_losses.append(loss.item())
                epoch_losses_gen.append(loss_generation.item())
                epoch_losses_vis.append(loss_visual.item())
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        # end of epoch
        avg_loss = np.mean(epoch_losses)
        avg_loss_gen = np.mean(epoch_losses_gen)
        avg_loss_vis = np.mean(epoch_losses_vis)
        std_loss = np.std(epoch_losses)
        print(f"Epoch {epoch}/{args.epochs} — Avg Loss: {avg_loss:.4f} ± {std_loss:.4f}")

        if args.do_wandb:
            # log epoch-level bar plots
            log_bar("loss_generation_per_timestep", loss_gen_by_t, epoch)
            log_bar("loss_visual_per_timestep",     loss_vis_by_t, epoch)
            log_bar("loss_total_per_timestep",      loss_tot_by_t, epoch)

            wandb.log({
                "train/epoch_loss":    avg_loss,
                "train/epoch_loss_gen":avg_loss_gen,
                "train/epoch_loss_vis":avg_loss_vis,
                "train/epoch_loss_std":std_loss,
                "train/epoch_lr":      optimizer.param_groups[0]['lr'],
                "epoch":               epoch
            }, step=global_step)
        # — now update sampling weights for next epoch —
        # build a tensor of per-t average total loss
        # fall back to the epoch‐wide average loss if a t was never sampled
        default = avg_loss  # global avg loss over all batches this epoch
        means = []
        for t in range(T):
            vals = loss_tot_by_t.get(t, [])
            if vals:
                means.append(np.mean(vals))
            else:
                means.append(default)
        avg_tot_losses = torch.tensor(means, dtype=torch.float32, device=device)
        avg_tot_losses = avg_tot_losses + 1e-13
        # normalize to sum to 1
        weights = avg_tot_losses / avg_tot_losses.sum()
        # checkpointing & early-stopping
        if avg_loss + es_min_delta < best_loss:
            if epoch!= 1:
                best_loss = avg_loss
            epochs_no_improve = 0
            ckpt_path = os.path.join(
                args.chkps_logging_path,
                f"diffusion_best_epoch{epoch}_loss{avg_loss:.4f}_{args.rec_importance}.pt"
            )
            torch.save(diffusion_model.state_dict(), ckpt_path)
            print(f"  ↳ New best model saved to {ckpt_path}")
            if args.do_wandb:
                print("Logging sample image...")
                diffusion_model.eval()
                sample_i(h, w, vae, diffusion_model,
                         torch.Generator(device=device),
                         epoch, global_step, device, seed=42)
                diffusion_model.train()
        else:
            epochs_no_improve += 1
            print(f"  ↳ No improvement for {epochs_no_improve} epoch(s)")

        if args.patience != -1 and epochs_no_improve >= args.patience:
            print(f"Stopping early after {epoch} epochs (patience {args.patience})")
            break

    # final save & sample
    final_path = os.path.join(
        args.chkps_logging_path,
        f"diffusion_final_epoch{epoch}_loss{avg_loss:.4f}_{args.rec_importance}.pt"
    )
    torch.save(diffusion_model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    if args.do_wandb:
        print("Logging final sample image...")
        diffusion_model.eval()
        sample_i(h, w, vae, diffusion_model,
                 torch.Generator(device=device),
                 epoch, global_step, device, seed=42)

    print("Training complete.")


if __name__ == '__main__':
    main()
