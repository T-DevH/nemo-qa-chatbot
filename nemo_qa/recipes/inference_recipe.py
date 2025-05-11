cat > nemo_qa/recipes/inference_recipe.py << 'EOF'
"""Inference recipe for NeMo QA Chatbot."""

import os
import logging
import json
from typing import Dict, Any, Optional
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

def load_model_for_inference(
    model_path: str,
    device: str = None
) -> MegatronGPTModel:
    """Load model for inference.
    
    Args:
        model_path: Path to the fine-tuned model
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading model from {model_path}")
    model = MegatronGPTModel.restore_from(model_path)
    model.to(device)
    model.eval()
    
    return model

def format_prompt(
    question: str,
    context: str = "",
    history: list = None
) -> str:
    """Format prompt for model input.
    
    Args:
        question: User question
        context: Optional context
        history: Optional conversation history
    
    Returns:
        Formatted prompt
    """
    if history is None:
        history = []
    
    # Format history
    formatted_history = ""
    for entry in history:
        formatted_history += f"Human: {entry[0]}\nAssistant: {entry[1]}\n\n"
    
    # Add context if provided
    context_str = f"Context: {context}\n\n" if context else ""
    
    # Format prompt
    prompt = f"{formatted_history}{context_str}Human: {question}\nAssistant:"
    
    return prompt

def export_model_for_nim(
    model_path: str,
    output_dir: str,
    config_path: Optional[str] = None
) -> str:
    """Export model for NeMo Inference Microservice.
    
    Args:
        model_path: Path to the fine-tuned model
        output_dir: Directory to save the exported model
        config_path: Optional path to NIM configuration
    
    Returns:
        Path to the exported model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load model
    model = load_model_for_inference(model_path)
    
    # Save model
    logger.info(f"Saving model to {model_dir}")
    model.save_to(model_dir)
    
    # Load or create NIM configuration
    nim_config = {
        "model_path": model_dir,
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
            nim_config.update(config)
    
    # Save NIM configuration
    config_file = os.path.join(output_dir, "nim_config.json")
    with open(config_file, "w") as f:
        json.dump(nim_config, f, indent=2)
    
    logger.info(f"Saved NIM configuration to {config_file}")
    
    return output_dir
EOF