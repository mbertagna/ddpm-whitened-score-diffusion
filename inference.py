import torch
from pathlib import Path
from torchvision.utils import save_image
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

def load_model_for_inference(checkpoint_path: Path, image_size: int, channels: int) -> GaussianDiffusion:
    """
    Loads a trained diffusion model from a checkpoint for inference.

    This function reconstructs the model architecture and loads the saved
    Exponential Moving Average (EMA) weights for higher quality generation.

    Parameters
   -------
    checkpoint_path : Path
        The file path to the saved model checkpoint (.pt file).
    image_size : int
        The size of the images the model was trained on (e.g., 32 for 32x32).
    channels : int
        The number of channels in the images (e.g., 1 for grayscale).

    Returns
   ----
    GaussianDiffusion
        The loaded diffusion model ready for inference.

    Raises
   ---
    FileNotFoundError
        If the checkpoint_path does not exist.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")
    
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4),
        channels=channels
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000
    )

    data = torch.load(checkpoint_path)
    
    # Load the EMA model state for better sample quality
    ema_model_state = data.get('ema')
    if ema_model_state is None:
        raise KeyError("Checkpoint does not contain 'ema' model state. Please use a checkpoint saved by the Trainer.")
        
    diffusion.load_state_dict(ema_model_state)
    return diffusion

def generate_and_save_samples(diffusion_model: GaussianDiffusion, batch_size: int, output_dir: Path):
    """
    Generates sample images using the diffusion model and saves them.

    Parameters
   -------
    diffusion_model : GaussianDiffusion
        The trained and loaded diffusion model.
    batch_size : int
        The number of images to generate.
    output_dir : Path
        The directory where the generated image grid will be saved.
    """
    print(f"Sampling {batch_size} new images...")
    
    # Generate a batch of images
    sampled_images = diffusion_model.sample(batch_size=batch_size)
    
    print(f"Generated images shape: {sampled_images.shape}")

    # Save the generated images as a single grid
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / 'sample.png'
    save_image(sampled_images, save_path)
    
    print(f"Saved generated images to {save_path}")

def main():
    """
    Main function to orchestrate the inference process.
    
    It defines the model configuration, loads the checkpoint, and
    generates sample images.
    """
    # Configuration
    MODEL_CHECKPOINT = Path('./results/model-10000.pt')
    IMAGE_SIZE = 32
    CHANNELS = 1
    BATCH_SIZE = 4
    OUTPUT_DIR = Path("./inference_results")

    try:
        # Load Model
        diffusion = load_model_for_inference(
            checkpoint_path=MODEL_CHECKPOINT,
            image_size=IMAGE_SIZE,
            channels=CHANNELS
        )
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        diffusion.to(device)

        # Generate Samples
        generate_and_save_samples(
            diffusion_model=diffusion,
            batch_size=BATCH_SIZE,
            output_dir=OUTPUT_DIR
        )

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
