import os
import torch
from dotenv import load_dotenv
from utils.logger import configure_logger

logger = configure_logger(__name__)

load_dotenv()

# Retrieve environment variables
hf_access_token = os.getenv("HF_ACCESS_TOKEN")
model = os.getenv("MODEL")
ckpt_path = os.getenv("CKPT_PATH")
default_height = int(os.getenv("DEFAULT_HEIGHT", 128))
default_width = int(os.getenv("DEFAULT_WIDTH", 128))
default_num_inference_steps = int(os.getenv("DEFAULT_NUM_INFERENCE_STEPS", 5))
default_seed = os.getenv("DEFAULT_SEED")
default_seed = int(default_seed) if default_seed and default_seed.isdigit() else None

# Add torch_dtype for dynamic data type configuration
torch_dtype = torch.bfloat16  

# Define output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

# Validate required environment variables
required_vars = {"HF_ACCESS_TOKEN": hf_access_token, "MODEL": model, "CKPT_PATH": ckpt_path}
missing_vars = [var for var, value in required_vars.items() if not value]

if missing_vars:
    logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. "
                 "Ensure your .env file is properly configured.")
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

if not ckpt_path.startswith("http"):
    logger.error(f"Invalid CKPT_PATH: {ckpt_path}. It must be a valid URL.")
    raise ValueError(f"Invalid CKPT_PATH: {ckpt_path}. It must be a valid URL.")

# Log successful configuration loading
logger.info("Configuration loaded successfully.")