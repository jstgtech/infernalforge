from utils.pipeline import initialize_pipeline
from utils.image_utils import process_image

def main():
    """Main entry point for the script."""
    try:
        # Initialize the pipeline (quantization is handled here)
        pipe = initialize_pipeline()

        # Define parameters for image generation
        prompt = "A (realistic) orange tabby cat with a wizard hat and a magic wand with stars in the background"
        height = 512
        width = 512
        num_inference_steps = 50
        seed = None  # Set to a specific value for reproducibility, or leave as None for random
        output_dir = "output"

        # Generate and save the image
        process_image(pipe, prompt, height, width, num_inference_steps, seed, output_dir)

    except Exception as e:
        # Log the exception
        from utils.logger import configure_logger
        logger = configure_logger(__name__)
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()