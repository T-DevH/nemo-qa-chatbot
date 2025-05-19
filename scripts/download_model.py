#!/usr/bin/env python3
"""Download LLAMA3 8B model from NVIDIA NGC."""
import os
import argparse
import logging
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def download_model(model_name="llama3-8b", output_dir="models/base", force=False):
    """Download the model from NVIDIA NGC.
    
    Args:
        model_name: Model name
        output_dir: Output directory
        force: Force download even if model exists
        
    Returns:
        Path to the downloaded model
    """
    # Model mapping
    models = {
        "llama3-8b": "nvidia/nemo-llama3-8b"
    }
    
    # Check if model exists
    model_path = models.get(model_name)
    if not model_path:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_name)
    
    # Check if model already exists
    if os.path.exists(model_dir) and os.listdir(model_dir) and not force:
        logger.info(f"Model already exists at {model_dir}. Use --force to redownload.")
        return model_dir
    
    # Download model
    logger.warning("This download requires sufficient GPU memory. Please ensure you have enough VRAM available.")
    logger.info(f"Downloading {model_name} from {model_path}...")
    
    try:
        model = MegatronGPTModel.from_pretrained(model_path)
        
        # Save model
        logger.info(f"Saving model to {model_dir}...")
        model.save_to(model_dir)
        logger.info(f"Model downloaded and saved to {model_dir}")
        
        return model_dir
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

def verify_model(model_dir):
    """Verify the downloaded model by loading it."""
    try:
        logger.info(f"Verifying model at {model_dir}...")
        model = MegatronGPTModel.restore_from(model_dir)
        logger.info("Model verification successful!")
        return True
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download LLAMA3 8B model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3-8b",
        choices=["llama3-8b"],
        help="Model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/base",
        help="Output directory"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if model exists"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the model after downloading"
    )
    
    args = parser.parse_args()
    model_dir = download_model(args.model_name, args.output_dir, args.force)
    
    if args.verify:
        verify_model(model_dir)

if __name__ == "__main__":
    main()