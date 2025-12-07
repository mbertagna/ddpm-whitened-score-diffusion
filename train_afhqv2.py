import torch
from pathlib import Path

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def train_afhqv2_diffusion(data_dir: Path):
    """
    Configure and train a diffusion model on the AFHQv2 dataset.

    The model is trained on RGB images downscaled to 64x64 resolution.
    The Trainer automatically handles loading images from subdirectories
    and resizing them to the specified image size.

    Parameters
    ----------
    data_dir : Path
        The directory containing the AFHQv2 training images.
        Expected structure: data_dir/{cat,dog,wild}/*.png
    """
    # Configure the U-Net model for RGB images (3 channels)
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=3
    )

    # Configure the Gaussian Diffusion model with 64x64 image size
    diffusion = GaussianDiffusion(
        model,
        image_size=64,
        timesteps=1000,
        sampling_timesteps=250
    )

    # Configure the Trainer
    trainer = Trainer(
        diffusion,
        str(data_dir),
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False
    )

    # Start the training process
    trainer.train()


def main():
    """
    Main function to run the AFHQv2 diffusion model training.
    """
    data_dir = Path("../../../data/AFHQv2/train")
    train_afhqv2_diffusion(data_dir)


if __name__ == "__main__":
    main()

