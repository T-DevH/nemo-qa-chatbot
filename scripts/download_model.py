cat > scripts/download_model.py << 'EOF'
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

def download_model(model_name="llama3-8b", output_dir="models/base"):
    """Download the model from NVIDIA NGC.
    
    Args:
        model_name: Model name
        output_dir: Output directory
    
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
    
    # Download model
    logger.info(f"Downloading {model_name} from {model_path}...")
    model = MegatronGPTModel.from_pretrained(model_path)
    
    # Save model
    logger.info(f"Saving model to {model_dir}...")
    model.save_to(model_dir)
    
    logger.info(f"Model downloaded and saved to {model_dir}")
    
    return model_dir

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
    args = parser.parse_args()
    
    download_model(args.model_name, args.output_dir)

if __name__ == "__main__":
    main()
EOF