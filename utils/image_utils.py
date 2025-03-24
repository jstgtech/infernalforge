import random
import datetime
import torch
import os
import re 
from utils.config import default_height, default_width, default_num_inference_steps, default_seed
from utils.logger import configure_logger
import time

logger = configure_logger(__name__)

def generate_image(
    pipe,
    prompt,
    height=None,
    width=None,
    num_inference_steps=None,
    seed=None
):
    """
    Generate an image using the provided pipeline.

    Args:
        pipe (FluxPipeline): The initialized pipeline object.
        prompt (str): The text prompt for image generation.
        height (int, optional): The height of the generated image. Defaults to environment or 128.
        width (int, optional): The width of the generated image. Defaults to environment or 128.
        num_inference_steps (int, optional): The number of inference steps. Defaults to environment or 5.
        seed (int, optional): The random seed for reproducibility. If None, a random seed will be generated.

    Returns:
        PIL.Image.Image: The generated image.
        int: The seed used for reproducibility.
    """
    # Use provided values or fall back to defaults
    height = height or default_height
    width = width or default_width
    num_inference_steps = num_inference_steps or default_num_inference_steps

    # Generate a random seed if none is provided
    if seed is None:
        seed = random.randint(0, 2**32 - 1)  # Generate a random seed in the range of 32-bit integers

    try:
        logger.info(f"Using seed: {seed}")
        # Set the seed for the generator
        generator = torch.Generator("cuda").manual_seed(seed)

        logger.info("Starting image generation...")
        start_time = time.time()  # Record the start time

        image = pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=3,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images[0]

        end_time = time.time()  # Record the end time
        time_taken = end_time - start_time  # Calculate the time taken
        logger.info(f"Image generation completed in {time_taken:.2f} seconds.")

        return image, seed  # Return the image and the seed
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise

def save_image(image, prompt, output_path):
    """
    Save the generated image to the output directory.

    Args:
        image (PIL.Image.Image): The generated image to save.
        prompt (str): The text prompt used to generate the image.
        output_path (str): The directory where the image will be saved.
    """
    try:
        # Ensure the output path is a directory
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Sanitize the prompt to create a safe filename
        sanitized_prompt = re.sub(r'[<>:"/\\|?*(){}]', '', prompt[:10])  # Remove invalid characters
        sanitized_prompt = sanitized_prompt.replace(' ', '_')  # Replace spaces with underscores

        # Construct the full file path
        filename = f'{sanitized_prompt}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
        full_path = os.path.join(output_path, filename)

        # Save the image
        image.save(full_path)
        logger.info(f"Image saved to {full_path}")
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        raise

def setup_output_directory(base_dir="output"):
    """
    Ensure the output directory exists.

    Args:
        base_dir (str): The base directory for saving images.

    Returns:
        str: The path to the output directory.
    """
    output_path = os.path.join(base_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

def process_image(pipe, prompt, height=512, width=512, num_inference_steps=10, seed=None, output_dir="output"):
    """
    Generate and save an image based on the given parameters.

    Args:
        pipe (FluxPipeline): The initialized pipeline object.
        prompt (str): The text prompt for image generation.
        height (int): The height of the generated image.
        width (int): The width of the generated image.
        num_inference_steps (int): The number of inference steps.
        seed (int, optional): The random seed for reproducibility.
        output_dir (str): The directory to save the generated image.

    Returns:
        str: The file path of the saved image.
        int: The seed used for reproducibility.
    """
    try:
        # Generate the image
        generated_image, seed = generate_image(pipe, prompt, height=height, width=width, num_inference_steps=num_inference_steps, seed=seed)

        # Ensure the output directory exists
        output_path = setup_output_directory(output_dir)

        # Save the image
        save_image(generated_image, prompt, output_path)

        # Get the actual saved file path
        sanitized_prompt = re.sub(r'[<>:"/\\|?*(){}]', '', prompt[:10])
        sanitized_prompt = sanitized_prompt.replace(' ', '_')
        filename = f'{sanitized_prompt}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
        image_path = os.path.join(output_path, filename)

        logger.info(f"Image generated and saved successfully with seed: {seed}")
        return image_path, seed
    except TimeoutError:
        logger.error("Image processing timed out")
        raise
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise