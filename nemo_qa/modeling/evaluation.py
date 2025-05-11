"""Evaluation utilities for NeMo QA Chatbot."""

import os
import logging
import json
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from tqdm import tqdm
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

def evaluate_model(
    model: MegatronGPTModel,
    test_data: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    batch_size: int = 1,
    max_length: int = 512,
    temperature: float = 0.0
) -> Dict[str, Any]:
    """Evaluate model.
    
    Args:
        model: Model to evaluate
        test_data: Test data
        output_path: Path to save evaluation results
        batch_size: Batch size
        max_length: Maximum length of generated text
        temperature: Sampling temperature
    
    Returns:
        Evaluation results
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    metrics = {
        "exact_match": [],
        "f1_score": [],
        "relevance_score": []
    }
    
    # Initialize results
    results = []
    
    # Evaluate model
    logger.info(f"Evaluating model on {len(test_data)} samples")
    
    for sample in tqdm(test_data):
        question = sample["question"]
        reference = sample["answer"]
        
        # Format prompt
        prompt = f"Human: {question}\nAssistant:"
        
        # Generate response
        with torch.inference_mode():
            output = model.generate(
                prompt,
                max_length=max_length,
                temperature=temperature
            )
        
        # Extract prediction
        prediction = output["text"][0].replace(prompt, "").strip()
        
        # Compute metrics
        exact_match = compute_exact_match(prediction, reference)
        f1 = compute_f1_score(prediction, reference)
        relevance_score = compute_relevance_score(question, prediction)
        
        metrics["exact_match"].append(exact_match)
        metrics["f1_score"].append(f1)
        metrics["relevance_score"].append(relevance_score)
        
        # Save result
        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "exact_match": exact_match,
            "f1_score": f1,
            "relevance_score": relevance_score
        })
    
    # Compute average metrics
    avg_metrics = {
        "exact_match": np.mean(metrics["exact_match"]),
        "f1_score": np.mean(metrics["f1_score"]),
        "relevance_score": np.mean(metrics["relevance_score"])
    }
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "metrics": avg_metrics,
                "results": results
            }, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    return avg_metrics

def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score.
    
    Args:
        prediction: Prediction
        reference: Reference
    
    Returns:
        Exact match score
    """
    return float(prediction.strip() == reference.strip())

def compute_f1_score(prediction: str, reference: str) -> float:
    """Compute F1 score.
    
    Args:
        prediction: Prediction
        reference: Reference
    
    Returns:
        F1 score
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    
    common = set(pred_tokens) & set(ref_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)

def compute_relevance_score(question: str, answer: str) -> float:
    """Compute relevance score.
    
    Args:
        question: Question
        answer: Answer
    
    Returns:
        Relevance score
    """
    # Placeholder implementation
    # In a real implementation, you would use a relevance model
    # to compute the relevance score
    
    return 0.8
EOF