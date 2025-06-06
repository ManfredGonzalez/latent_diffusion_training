# Latent Diffusion for Aerial Pineapple Generation
## create a conda env:
```bash
conda create -n ddpm_training python=3.10

conda activate ddpm_training

pip install numpy tqdm pillow pyYaml
```
Linux:
```bash
pip3 install torch torchvision torchaudio
```
Windows:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
```bash
pip install opencv-python
pip install wandb
pip install wandb[media]
```
## Steps to clone this repo:

1. Clone it in an empty directory with the following command:
git clone --recurse-submodules <your-repo-URL>
2. Download the pretrained weights of the VAE from here: https://drive.google.com/drive/folders/18zJVWhoJJV-mUOUop2Ev_Yqc_J_T2wDZ?usp=drive_link
3. how to run it for training
python train.py --dataset_path path/to/your/dataset --vae_chkp path/to/your/checkpoint/file.pt