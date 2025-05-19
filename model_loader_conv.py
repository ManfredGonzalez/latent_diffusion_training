import os
import sys
import torch
from diffusion_conv import Diffusion
#get the current working directory
current_dir = os.getcwd()
path_to_add = os.path.join(current_dir,"VAE_training")
# check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from VAE_training.VAE import VAE
def load_models(diffusion_model_path, vae_model_path, device, idle_device=None):
    # instantiate the architecture
    diffusion = Diffusion().to(device)

    # load the raw state dict
    ckpt = torch.load(diffusion_model_path, map_location=device)
    # if it’s wrapped in another dict, unwrap it:
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    diffusion.load_state_dict(state_dict)

    # now diffusion is a Module and you can call .eval()
    diffusion.eval()

    # load VAE exactly the same way you already are
    vae = VAE().to(device)
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.eval()

    if idle_device:
        diffusion.to(idle_device)
        vae.to(idle_device)

    # freeze
    for p in diffusion.parameters(): p.requires_grad = False
    for p in vae.parameters():       p.requires_grad = False

    return diffusion, vae