"""LoRA implementation utilities for NeMo QA Chatbot."""

import logging
from typing import Dict, Any, List, Optional
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    modules_to_save: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Create LoRA configuration.
    
    Args:
        r: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: List of target modules
        bias: Bias type
        modules_to_save: List of modules to save
    
    Returns:
        LoRA configuration
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    if modules_to_save is None:
        modules_to_save = []
    
    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
        "bias": bias,
        "modules_to_save": modules_to_save
    }

def enable_lora_for_model(
    model: MegatronGPTModel,
    lora_config: Dict[str, Any]
) -> MegatronGPTModel:
    """Enable LoRA for model.
    
    Args:
        model: Model to enable LoRA for
        lora_config: LoRA configuration
    
    Returns:
        Model with LoRA enabled
    """
    logger.info(f"Enabling LoRA with config: {lora_config}")
    model.enable_lora(**lora_config)
    
    return model

def merge_lora_weights(model: MegatronGPTModel) -> MegatronGPTModel:
    """Merge LoRA weights into base model.
    
    Args:
        model: Model with LoRA
    
    Returns:
        Model with merged weights
    """
    logger.info("Merging LoRA weights")
    model.merge_lora_weights()
    
    return model
