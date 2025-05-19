#!/usr/bin/env python3
"""Download LLAMA3 8B model from NVIDIA NGC."""
import os
import argparse
import logging
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import torch
from tqdm import tqdm
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from huggingface_hub import HfFolder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def get_ngc_token() -> Optional[str]:
    """Get NGC token from environment or user input."""
    token = os.getenv("NGC_TOKEN")
    if not token:
        token = HfFolder.get_token()
    if not token:
        logger.warning("No NGC token found. Please set NGC_TOKEN environment variable or login to HuggingFace.")
        return None
    return token

def calculate_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def verify_disk_space(path: str, required_space_gb: float = 20) -> bool:
    """Verify if there's enough disk space."""
    free_space = shutil.disk_usage(path).free
    required_space = required_space_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    return free_space >= required_space

def download_model(
    model_name: str = "llama3-8b",
    output_dir: str = "models/base",
    force: bool = False,
    verify: bool = True
) -> str:
    """Download the model from NVIDIA NGC.
    
    Args:
        model_name: Model name
        output_dir: Output directory
        force: Force download even if model exists
        verify: Verify model after download
        
    Returns:
        Path to the downloaded model
    """
    # Model mapping with checksums
    models: Dict[str, Dict[str, Any]] = {
        "llama3-8b": {
            "path": "nvidia/nemo-llama3-8b",
            "size_gb": 16,
            "expected_checksum": None  # To be added when available
        }
    }
    
    # Check if model exists
    model_info = models.get(model_name)
    if not model_info:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_dir = output_path / model_name
    
    # Check if model already exists
    if model_dir.exists() and list(model_dir.glob("*")) and not force:
        logger.info(f"Model already exists at {model_dir}. Use --force to redownload.")
        return str(model_dir)
    
    # Verify disk space
    if not verify_disk_space(str(output_path), model_info["size_gb"]):
        raise OSError(f"Not enough disk space. Required: {model_info['size_gb']}GB")
    
    # Check NGC token
    token = get_ngc_token()
    if not token:
        raise ValueError("NGC token not found. Please set NGC_TOKEN environment variable.")
    
    # Download model
    logger.warning("This download requires sufficient GPU memory. Please ensure you have enough VRAM available.")
    logger.info(f"Downloading {model_name} from {model_info['path']}...")
    
    try:
        with torch.cuda.device(0):  # Ensure we're using the first GPU
            model = MegatronGPTModel.from_pretrained(
                model_info["path"],
                token=token,
                progress_bar=True
            )
            
            # Save model
            logger.info(f"Saving model to {model_dir}...")
            model.save_to(str(model_dir))
            
            # Cleanup
            del model
            torch.cuda.empty_cache()
            
            if verify:
                if not verify_model(str(model_dir)):
                    raise RuntimeError("Model verification failed")
            
            logger.info(f"Model downloaded and saved to {model_dir}")
            return str(model_dir)
            
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        raise

def verify_model(model_dir: str) -> bool:
    """Verify the downloaded model by loading it and checking its structure."""
    try:
        logger.info(f"Verifying model at {model_dir}...")
        
        # Check if all required files exist
        required_files = ["model_config.yaml", "model_weights.ckpt"]
        for file in required_files:
            if not (Path(model_dir) / file).exists():
                logger.error(f"Required file {file} not found")
                return False
        
        # Load model
        model = MegatronGPTModel.restore_from(model_dir)
        
        # Verify model architecture
        if not hasattr(model, "config"):
            logger.error("Model missing config attribute")
            return False
        
        # Verify model parameters
        if not hasattr(model, "parameters"):
            logger.error("Model missing parameters")
            return False
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
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
    parser.add_argument(
        "--skip_verification",
        action="store_true",
        help="Skip model verification after download"
    )
    
    args = parser.parse_args()
    
    try:
        model_dir = download_model(
            args.model_name,
            args.output_dir,
            args.force,
            not args.skip_verification
        )
        logger.info(f"Model successfully downloaded to: {model_dir}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()