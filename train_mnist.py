import torch
from torchvision.datasets import MNIST
from pathlib import Path

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

def download_and_save_mnist(data_dir: Path):
    """
    Downloads the MNIST dataset and saves it as PNG images in a directory.

    If the directory already exists, this function does nothing.

    Parameters
    ----------
    data_dir : Path
        The directory where the MNIST images will be saved.
    """
    if not data_dir.exists():
        print("Downloading and saving MNIST images...")
        data_dir.mkdir(exist_ok=True, parents=True)
        mnist_dataset = MNIST(root='./data', train=True, download=True)
        
        for i, (image, label) in enumerate(mnist_dataset):
            image.save(data_dir / f"mnist_train_{i}.png")
        print(f"Saved {len(mnist_dataset)} MNIST images to {data_dir}")
    else:
        print("MNIST images folder already exists.")

def train_mnist_diffusion(data_dir: Path):
    """
    Configures and trains a diffusion model on the MNIST dataset.

    Parameters
    ----------
    data_dir : Path
        The directory containing the MNIST PNG images.
    """
    # Configure the U-Net model for single-channel (grayscale) images
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4),
        channels=1
    )

    # Configure the Gaussian Diffusion model
    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,
        sampling_timesteps=250
    )

    # Configure the Trainer
    trainer = Trainer(
        diffusion,
        str(data_dir),
        train_batch_size=32,
        train_lr=8e-5,
        train_num_steps=10000,
        gradient_accumulate_every=2,
        ema_decay=0.995,
        amp=True,
        calculate_fid=False
    )

    # Start the training process
    trainer.train()

def main():
    """
    Main function to run the MNIST download and training process.
    """
    data_dir = Path("./mnist_images")
    download_and_save_mnist(data_dir)
    train_mnist_diffusion(data_dir)

if __name__ == "__main__":
    main()
