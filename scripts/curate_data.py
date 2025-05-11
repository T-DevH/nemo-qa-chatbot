#!/usr/bin/env python3
"""Data curation pipeline for NeMo QA Chatbot."""

import os
import argparse
import logging
from nemo_qa.recipes.curator_recipe import curator_recipe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Curate data for Q&A fine-tuning")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with raw documents"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/base/llama3-8b",
        help="Path to the pretrained model"
    )
    args = parser.parse_args()
    
    # Run curator recipe
    logger.info("Starting data curation...")
    stats = curator_recipe(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path
    )
    
    logger.info("Data curation complete!")
    logger.info(f"Processed {stats['documents']['input_files']} documents")
    logger.info(f"Generated {stats['qa_pairs']} Q&A pairs")
    logger.info(f"Filtered to {stats['filtered_pairs']} high-quality pairs")
    logger.info(f"Training set: {stats['train_pairs']} pairs")
    logger.info(f"Validation set: {stats['val_pairs']} pairs")
    logger.info(f"Test set: {stats['test_pairs']} pairs")

if __name__ == "__main__":
    main()
