#!/usr/bin/env python3
"""Training script for NeMo QA Chatbot."""

import os
import argparse
import logging
from nemo_qa.recipes.lora_recipe import lora_finetuning_recipe
from nemo_qa.config.lora_config import LoRAConfig
from nemo_qa.config.training_config import TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to base model"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        required=True,
        help="Path to validation data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the model"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=3,
        help="Maximum number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    args = parser.parse_args()
    
    # Create LoRA config
    lora_config = LoRAConfig(
        r=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    ).to_dict()
    
    # Create training config
    training_config = TrainingConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        devices=args.devices
    ).to_dict()
    
    # Train model
    logger.info("Starting LoRA fine-tuning...")
    model_path = lora_finetuning_recipe(
        model_path=args.model_path,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_dir=args.output_dir,
        lora_config=lora_config,
        training_args=training_config
    )
    
    logger.info(f"Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()

# This file is intended for training.
# Add your training code here.
