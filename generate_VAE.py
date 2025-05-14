
from Dataset import PineappleDataset
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
#get the current working directory
current_dir = os.getcwd()
path_to_add = os.path.join(current_dir,"VAE_training")
# check if the path is already in sys.path
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

from VAE_training.VAE import VAE
# 1. Load the trained VAE checkpoint
vae = VAE()
checkpoint_path = "checkpoints/vae/betaKL@1.0/weights_ck_398.pt"  # Replace with your checkpoint file
checkpoint = torch.load(checkpoint_path,weights_only=True)
vae.load_state_dict(checkpoint)
vae.eval()
vae.cuda()  # Move model to GPU if available

index = 3
test_set = PineappleDataset(train=False, train_ratio=0.8, dataset_path="/home/tico/Desktop/master_research/fabian_research/VAE_training/FULL_VERTICAL_PINEAPPLE/FULL_UNIFIED/*")
# 2. Load the test image
test_image = test_set[index]['image']
test_image = torch.tensor(test_image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
# 3. Pass the image through the model
with torch.no_grad():
    #reconstructed_image, _, _ = model(test_image)
    batch_size, _, height, width = test_image.shape
    # The encoder expects noise with shape (Batch_Size, 4, Height/8, Width/8).
    noise = torch.randn((batch_size, 4, height // 8, width // 8), device=test_image.device)
    latent, mean, logvar = vae.encoder(test_image, noise)
    reconstructed_image = vae.decoder(latent)
    reconstructed_image = reconstructed_image.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))  # Change to HWC format
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)  # Convert to uint8
# 4. Display the original and reconstructed images
fig = plt.figure(figsize=(12, 6))
columns = 2
rows = 1
original_image = test_set[index]['image'].transpose((1, 2, 0))
original_image = (original_image * 255).astype(np.uint8)  # Convert to uint8
# add labels to the images
fig.add_subplot(rows, columns, 1, title='Original Image')
plt.imshow(original_image)
fig.add_subplot(rows, columns, 2, title='Reconstructed Image')
plt.imshow(reconstructed_image)
plt.savefig("recontructed_image.pdf", format='pdf', bbox_inches='tight')
plt.close()
