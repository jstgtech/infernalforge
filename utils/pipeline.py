import torch
import gc
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
from huggingface_hub import hf_hub_download
from utils.config import model, ckpt_path, hf_access_token, torch_dtype
from utils.logger import configure_logger

logger = configure_logger(__name__)

def initialize_pipeline():
    """
    Initialize the FluxPipeline with explicit transformer initialization and quantization.

    Returns:
        FluxPipeline: The initialized and quantized pipeline.
    """
    logger.info("Initializing the pipeline...")
    try:
        gc.collect()
        torch.cuda.empty_cache()

        # Initialize the transformer with quantization
        logger.info("Initializing the transformer with quantization...")
        transformer = FluxTransformer2DModel.from_single_file(
            ckpt_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            torch_dtype=torch_dtype,
        )
        logger.info("Transformer initialized and quantized successfully.")

        # Initialize the pipeline with the preloaded transformer
        pipe = FluxPipeline.from_pretrained(
            model,
            transformer=transformer,
            torch_dtype=torch_dtype,
            token=hf_access_token
        )

        # Enable model CPU offload for memory efficiency
        pipe.enable_model_cpu_offload()

        logger.info("Pipeline initialized successfully.")
        return pipe
    except Exception as e:
        logger.error(f"Failed to initialize the pipeline: {e}")
        raise
