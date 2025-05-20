#!/usr/bin/env python3
"""Download LLAMA3 8B model using NGC CLI."""
import os
import argparse
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def verify_disk_space(path: str, required_space_gb: float = 20) -> bool:
    """Verify if there's enough disk space."""
    free_space = shutil.disk_usage(path).free
    required_space = required_space_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    return free_space >= required_space

def download_model(
    model_name: str = "llama3-8b",
    output_dir: str = "models/base",
    force: bool = False,
) -> str:
    """Download the model from NVIDIA NGC.
    
    Args:
        model_name: Model name
        output_dir: Output directory
        force: Force download even if model exists
        
    Returns:
        Path to the downloaded model
    """
    # Model mapping to NGC model paths
    models: Dict[str, Dict[str, Any]] = {
        "llama3-8b": {
            "ngc_path": "nvidia/nemo/llama-3_1-8b-nemo:1.0",
            "size_gb": 16,
        },
        "llama3.1-8b": {
            "ngc_path": "nvidia/nemo/llama-3_1-8b-nemo:1.0",
            "size_gb": 16,
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
    
    # Download model using NGC CLI
    logger.warning("This download requires sufficient disk space. Please ensure you have enough space available.")
    logger.info(f"Downloading {model_name} from NGC: {model_info['ngc_path']}...")
    
    try:
        # Create the output directory with appropriate permissions
        model_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(model_dir, 0o755)  # Set read/write/execute permissions for user, read/execute for others
        
        # Use NGC CLI to download the model (with full path)
        cmd = [
            "/data/TAO/getting_started_v4.0.0/setup/ngc-cli/ngc",  # Full path to NGC CLI
            "registry",
            "model",
            "download-version",
            model_info["ngc_path"],
            "--dest",
            str(model_dir)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream the output to show download progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
                
        # Get the return code
        return_code = process.poll()
        
        # Get any error output
        error_output = process.stderr.read()
        if error_output:
            logger.warning(f"Error output: {error_output}")
        
        # Check if download was successful
        if return_code == 0:
            logger.info(f"Model successfully downloaded to {model_dir}")
            return str(model_dir)
        else:
            raise RuntimeError(f"NGC CLI command failed with return code {return_code}")
            
    except FileNotFoundError:
        logger.error("NGC CLI not found at the specified path. Please check the path.")
        logger.info("Path used: /data/TAO/getting_started_v4.0.0/setup/ngc-cli/ngc")
        raise
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if model_dir.exists():
            shutil.rmtree(model_dir)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download LLAMA3 8B model from NGC")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3-8b",
        choices=["llama3-8b", "llama3.1-8b"],
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
    
    args = parser.parse_args()
    
    try:
        model_dir = download_model(
            args.model_name,
            args.output_dir,
            args.force
        )
        logger.info(f"Model successfully downloaded to: {model_dir}")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()