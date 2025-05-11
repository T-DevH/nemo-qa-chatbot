#!/usr/bin/env python3
"""Evaluation script for NeMo QA Chatbot."""

import os
import argparse
import logging
import json
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from tqdm import tqdm
from typing import List, Dict, Any
import numpy as np
from nemo_qa.modeling.evaluation import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to the test data"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the evaluation results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate"
    )
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = MegatronGPTModel.restore_from(args.model_path)
    model.eval()
    
    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    test_data = []
    with open(args.test_data, "r") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    if args.max_samples and args.max_samples < len(test_data):
        test_data = test_data[:args.max_samples]
    
    # Evaluate model
    logger.info(f"Evaluating model on {len(test_data)} samples")
    metrics = evaluate_model(
        model=model,
        test_data=test_data,
        output_path=args.output_path,
        batch_size=1,
        max_length=512,
        temperature=0.0
    )
    
    # Print metrics
    logger.info("Evaluation results:")
    logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Relevance Score: {metrics['relevance_score']:.4f}")
    
    if args.output_path:
        logger.info(f"Evaluation results saved to {args.output_path}")

# This file is intended for evaluation purposes.
# Add your evaluation code here.

if __name__ == "__main__":
    main()
