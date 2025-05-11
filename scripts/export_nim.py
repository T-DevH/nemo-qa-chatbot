#!/usr/bin/env python3
"""Export model as NeMo Inference Microservice."""

import os
import argparse
import logging
import shutil
from nemo_qa.recipes.inference_recipe import export_model_for_nim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Export model as NIM")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save NIM"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to NIM configuration"
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export model for NIM
    logger.info("Exporting model for NIM...")
    nim_path = export_model_for_nim(
        model_path=args.model_path,
        output_dir=args.output_dir,
        config_path=args.config_path
    )
    
    # Copy NIM implementation
    logger.info("Copying NIM implementation...")
    nim_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nim")
    nim_dst = os.path.join(args.output_dir, "nim")
    shutil.copytree(nim_src, nim_dst, dirs_exist_ok=True)
    
    logger.info(f"NIM export complete! NIM saved to {nim_path}")
    logger.info("To build the NIM container:")
    logger.info(f"cd {args.output_dir} && docker build -t llama3-qa-chatbot-nim .")

if __name__ == "__main__":
    main()

# This file is intended for exporting NIM.
# Add your export code here.
