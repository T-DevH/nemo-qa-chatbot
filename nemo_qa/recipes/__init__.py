"""NeMo 2.0 recipes for NeMo QA Chatbot.

This module provides recipe-based workflows for data curation, LoRA fine-tuning, and inference.
These recipes are designed to be modular, reusable components for building end-to-end pipelines.
"""

from nemo_qa.recipes.curator_recipe import curator_recipe
from nemo_qa.recipes.lora_recipe import create_lora_config, setup_lora_finetuning, lora_finetuning_recipe
from nemo_qa.recipes.inference_recipe import load_model_for_inference, format_prompt, export_model_for_nim

__all__ = [
    "curator_recipe",
    "create_lora_config",
    "setup_lora_finetuning",
    "lora_finetuning_recipe",
    "load_model_for_inference",
    "format_prompt",
    "export_model_for_nim"
]
