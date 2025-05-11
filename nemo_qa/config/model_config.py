cat > nemo_qa/config/model_config.py << 'EOF'
"""LLAMA3 8B model configuration for NeMo QA Chatbot.

This module defines the configuration for the LLAMA3 8B model, including model paths,
precision, sequence lengths, batch sizes, and various model-specific parameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

@dataclass
class LLAMA3Config:
    """Configuration for LLAMA3 8B model.
    
    This class defines the configuration parameters for the LLAMA3 8B model,
    including paths, precision, sequence lengths, and model-specific parameters.
    
    Attributes:
        model_name: Name of the model.
        pretrained_path: Path to the pretrained model.
        precision: Precision for training and inference (bf16, fp16, fp32).
        max_seq_length: Maximum sequence length for input.
        micro_batch_size: Batch size per GPU.
        global_batch_size: Total batch size across all GPUs.
        tensor_model_parallel_size: Number of GPUs for tensor parallelism.
        use_flash_attention: Whether to use flash attention for faster inference.
        attention_dropout: Dropout rate for attention layers.
        hidden_dropout: Dropout rate for hidden layers.
        vocab_size: Vocabulary size.
        num_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        hidden_size: Size of hidden layers.
        ffn_hidden_size: Size of feed-forward network hidden layers.
        kv_channels: Size of key and value projections.
        init_method_std: Standard deviation for parameter initialization.
        layernorm_epsilon: Epsilon for layer normalization.
        use_cache: Whether to use KV cache for faster inference.
        rope_theta: RoPE theta parameter.
        rope_scaling: RoPE scaling configuration.
        activation: Activation function to use.
    """
    
    # Basic configuration
    model_name: str = "llama3-8b"
    pretrained_path: str = "nvidia/nemo-llama3-8b"
    precision: str = "bf16"
    
    # Sequence and batch size configuration
    max_seq_length: int = 2048
    micro_batch_size: int = 4
    global_batch_size: int = 32
    
    # Parallelism configuration
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    
    # Attention and dropout configuration
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    # Model architecture configuration
    vocab_size: int = 32000
    num_layers: int = 32  # LLAMA3 8B has 32 layers
    num_attention_heads: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 11008  # 4096 * 2.7 (for LLAMA3)
    kv_channels: int = 128
    
    # Initialization and normalization
    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    
    # Inference configuration
    use_cache: bool = True
    
    # RoPE configuration
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    # Activation function
    activation: str = "silu"
    
    # LLM-specific options
    apply_residual_connection_post_layernorm: bool = False
    
    # Environment variables for loading the model
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        if self.rope_scaling is None:
            self.rope_scaling = {"type": "linear", "factor": 2.0}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, config_path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration.
        """
        import json
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> "LLAMA3Config":
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Loaded configuration.
        """
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_nemo_config(self) -> Dict[str, Any]:
        """Get NeMo configuration dictionary.
        
        Returns:
            NeMo configuration dictionary.
        """
        config = self.to_dict()
        # Convert to NeMo-specific configuration format
        nemo_config = {
            "model": {
                "name": config["model_name"],
                "pretrained_path": config["pretrained_path"],
                "precision": config["precision"],
                "tensor_model_parallel_size": config["tensor_model_parallel_size"],
                "pipeline_model_parallel_size": config["pipeline_model_parallel_size"],
                "vocab_size": config["vocab_size"],
                "num_layers": config["num_layers"],
                "num_attention_heads": config["num_attention_heads"],
                "hidden_size": config["hidden_size"],
                "ffn_hidden_size": config["ffn_hidden_size"],
                "kv_channels": config["kv_channels"],
                "max_position_embeddings": config["max_seq_length"],
                "use_flash_attention": config["use_flash_attention"],
                "attention_dropout": config["attention_dropout"],
                "hidden_dropout": config["hidden_dropout"],
                "init_method_std": config["init_method_std"],
                "layernorm_epsilon": config["layernorm_epsilon"],
                "use_cache": config["use_cache"],
                "activation": config["activation"],
                "apply_residual_connection_post_layernorm": config["apply_residual_connection_post_layernorm"],
                "rotary_percentage": 1.0,
                "rotary_embedding_base": config["rope_theta"],
                "rotary_embedding_scaling": config["rope_scaling"],
            },
            "trainer": {
                "devices": config["tensor_model_parallel_size"],
                "num_nodes": 1,
                "precision": config["precision"],
                "accelerator": "gpu",
                "micro_batch_size": config["micro_batch_size"],
                "global_batch_size": config["global_batch_size"],
            }
        }
        return nemo_config

# Example usage
if __name__ == "__main__":
    # Create a default configuration
    config = LLAMA3Config()
    
    # Print the configuration
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save the configuration
    config.save("llama3_config.json")
    
    # Load the configuration
    loaded_config = LLAMA3Config.load("llama3_config.json")
    
    # Get NeMo configuration
    nemo_config = config.get_nemo_config()
EOF