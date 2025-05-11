cat > nemo_qa/config/lora_config.py << 'EOF'
"""LoRA configuration for NeMo QA Chatbot.

This module defines the configuration for Low-Rank Adaptation (LoRA) fine-tuning,
including rank, scaling factor, dropout, and target modules.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.
    
    This class defines the configuration parameters for Low-Rank Adaptation (LoRA) fine-tuning,
    including rank, scaling factor, dropout, and target modules.
    
    Attributes:
        r: Rank of low-rank matrices.
        alpha: Scaling factor for LoRA updates.
        dropout: Dropout probability for LoRA layers.
        target_modules: List of module names to apply LoRA to.
        bias: Bias type ("none", "all", or "lora_only").
        modules_to_save: List of module names to save in addition to LoRA parameters.
        task_type: Task type for LoRA.
        adapter_name: Name of the adapter.
        init_weights: Whether to initialize weights.
        inference_mode: Whether to use inference mode.
        scaling_factor: Scaling factor for merged weights.
        fan_in_fan_out: Whether target modules have a fan-in/fan-out structure.
    """
    
    # LoRA parameters
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"
    
    # Additional parameters
    modules_to_save: List[str] = field(default_factory=list)
    task_type: Optional[str] = "CAUSAL_LM"
    adapter_name: str = "default"
    init_weights: bool = True
    inference_mode: bool = False
    scaling_factor: float = 1.0
    fan_in_fan_out: bool = False
    
    # Implementation-specific parameters
    use_rslora: bool = False  # Whether to use Rank-Stabilized LoRA
    use_tucker_lora: bool = False  # Whether to use Tucker-LoRA
    use_lokr: bool = False  # Whether to use Low-rank Kronecker adaptation
    
    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
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
    def load(cls, config_path: str) -> "LoRAConfig":
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
    
    def get_peft_config(self) -> Dict[str, Any]:
        """Get PEFT configuration dictionary for compatibility with HuggingFace PEFT.
        
        Returns:
            PEFT configuration dictionary.
        """
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type,
            "adapter_name": self.adapter_name,
            "init_lora_weights": self.init_weights,
            "inference_mode": self.inference_mode,
            "scaling_factor": self.scaling_factor,
            "fan_in_fan_out": self.fan_in_fan_out,
        }
    
    def get_nemo_lora_config(self) -> Dict[str, Any]:
        """Get NeMo LoRA configuration dictionary.
        
        Returns:
            NeMo LoRA configuration dictionary.
        """
        config = {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type,
            "adapter_name": self.adapter_name,
            "init_weights": self.init_weights,
            "inference_mode": self.inference_mode,
            "scaling_factor": self.scaling_factor,
            "fan_in_fan_out": self.fan_in_fan_out,
        }
        
        # Add implementation-specific parameters
        if self.use_rslora:
            config["use_rslora"] = True
        
        if self.use_tucker_lora:
            config["use_tucker_lora"] = True
        
        if self.use_lokr:
            config["use_lokr"] = True
        
        return config

# Example usage
if __name__ == "__main__":
    # Create a default configuration
    config = LoRAConfig()
    
    # Print the configuration
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save the configuration
    config.save("lora_config.json")
    
    # Load the configuration
    loaded_config = LoRAConfig.load("lora_config.json")
    
    # Get PEFT configuration
    peft_config = config.get_peft_config()
    
    # Get NeMo LoRA configuration
    nemo_config = config.get_nemo_lora_config()
EOF