"""Main model implementation for NeMo QA Chatbot."""

import os
import logging
import torch
from typing import Dict, Any, Optional
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

class QAChatbotModel:
    """Q&A Chatbot model."""
    
    def __init__(
        self,
        model_path: str,
        device: str = None
    ):
        """Initialize model.
        
        Args:
            model_path: Path to the model
            device: Device to load the model on
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = MegatronGPTModel.restore_from(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional keyword arguments
        
        Returns:
            Generated response
        """
        with torch.inference_mode():
            output = self.model.generate(
                prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                **kwargs
            )
        
        return output
    
    def save(self, save_path: str):
        """Save model.
        
        Args:
            save_path: Path to save the model
        """
        logger.info(f"Saving model to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_to(save_path)

def load_model(model_path: str, device: str = None) -> QAChatbotModel:
    """Load model.
    
    Args:
        model_path: Path to the model
        device: Device to load the model on
    
    Returns:
        Loaded model
    """
    return QAChatbotModel(model_path, device)
