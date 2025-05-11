cat > nim/model_handler.py << 'EOF'
"""NIM model handler implementation."""

import torch
import logging
from typing import Dict, Any, List
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

class LLAMA3QAChatbotHandler:
    """Model handler for LLAMA3 Q&A Chatbot."""
    
    def __init__(self, **kwargs):
        """Initialize the model handler.
        
        Args:
            **kwargs: Keyword arguments
        """
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = kwargs.get("model_path")
        self.max_length = kwargs.get("max_length", 512)
        self.temperature = kwargs.get("temperature", 0.7)
        self.top_p = kwargs.get("top_p", 0.9)
        self.top_k = kwargs.get("top_k", 50)
        
        logger.info(f"Initialized model handler with config: {kwargs}")
    
    def initialize(self, artifacts_dir: str) -> None:
        """Load the model from artifacts directory.
        
        Args:
            artifacts_dir: Path to the artifacts directory
        """
        logger.info(f"Loading model from {artifacts_dir}")
        self.model = MegatronGPTModel.restore_from(self.model_path or artifacts_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
    
    def preprocess(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the request.
        
        Args:
            request: Request data
        
        Returns:
            Preprocessed data
        """
        question = request.get("question", "")
        context = request.get("context", "")
        history = request.get("history", [])
        
        # Format prompt with history and question
        prompt = self._format_prompt(question, context, history)
        
        return {"prompt": prompt, "meta": request.get("meta", {})}
    
    def _format_prompt(self, question: str, context: str, history: List) -> str:
        """Format the prompt with history and context.
        
        Args:
            question: User question
            context: Context
            history: Conversation history
        
        Returns:
            Formatted prompt
        """
        formatted_history = ""
        for entry in history:
            formatted_history += f"Human: {entry[0]}\nAssistant: {entry[1]}\n\n"
        
        context_str = f"\nContext: {context}\n\n" if context else ""
        
        return f"{formatted_history}{context_str}Human: {question}\nAssistant:"
    
    def inference(self, model_input: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the preprocessed input.
        
        Args:
            model_input: Preprocessed input
        
        Returns:
            Inference output
        """
        prompt = model_input["prompt"]
        
        logger.info(f"Running inference on prompt: {prompt[:50]}...")
        
        with torch.inference_mode():
            output = self.model.generate(
                prompt,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=1.2,
                return_attention=True
            )
        
        # Extract generated text and attention (for explainability)
        text = output["text"][0]
        attention = output.get("attention_weights", None)
        
        return {
            "text": text,
            "attention": attention,
            "meta": model_input.get("meta", {})
        }
    
    def postprocess(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocess the model outputs.
        
        Args:
            inference_output: Inference output
        
        Returns:
            Postprocessed output
        """
        # Extract the assistant's response (without the prompt)
        text = inference_output["text"]
        attention = inference_output.get("attention")
        
        # Process attention for explainability features
        explainability_data = self._process_attention(attention) if attention is not None else None
        
        return {
            "response": text,
            "explainability": explainability_data,
            "meta": inference_output.get("meta", {})
        }
    
    def _process_attention(self, attention):
        """Process attention weights for explainability.
        
        Args:
            attention: Attention weights
        
        Returns:
            Processed attention data
        """
        if attention is None:
            return None
        
        try:
            # Simplify attention for visualization
            # This is a placeholder implementation
            mean_attention = attention.mean(dim=0).tolist()
            
            return {
                "attention_scores": mean_attention
            }
        except Exception as e:
            logger.error(f"Error processing attention weights: {e}")
            return None
EOF