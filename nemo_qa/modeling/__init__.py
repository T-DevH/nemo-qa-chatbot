"""Model implementation for NeMo QA Chatbot.

This module provides utilities for working with NeMo models, including loading models,
applying LoRA fine-tuning, and evaluating models.
"""

from nemo_qa.modeling.model import QAChatbotModel, load_model
from nemo_qa.modeling.lora import create_lora_config, enable_lora_for_model, merge_lora_weights
from nemo_qa.modeling.evaluation import evaluate_model, compute_exact_match, compute_f1_score

__all__ = [
    "QAChatbotModel",
    "load_model",
    "create_lora_config",
    "enable_lora_for_model",
    "merge_lora_weights",
    "evaluate_model",
    "compute_exact_match",
    "compute_f1_score"
]
