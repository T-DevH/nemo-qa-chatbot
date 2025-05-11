# Create config/__init__.py
cat > nemo_qa/config/__init__.py << 'EOF'
"""Configuration for NeMo QA Chatbot.


This module provides configuration classes for different aspects of the NeMo QA Chatbot:
- Model configuration (LLAMA3Config)
- Training configuration (TrainingConfig)
- LoRA configuration (LoRAConfig)
- NeMo Inference Microservice configuration (NIMConfig)
"""

from nemo_qa.config.model_config import LLAMA3Config
from nemo_qa.config.training_config import TrainingConfig
from nemo_qa.config.lora_config import LoRAConfig
from nemo_qa.config.nim_config import NIMConfig

__all__ = [
    "LLAMA3Config",
    "TrainingConfig",
    "LoRAConfig",
    "NIMConfig",
]
EOF

# Create model_config.py
cat > nemo_qa/config/model_config.py << 'EOF'
"""LLAMA3 8B model configuration."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LLAMA3Config:
    """Configuration for LLAMA3 8B model."""
    
    model_name: str = "llama3-8b"
    pretrained_path: str = "nvidia/nemo-llama3-8b"
    precision: str = "bf16"
    max_seq_length: int = 2048
    micro_batch_size: int = 4
    global_batch_size: int = 32
    tensor_model_parallel_size: int = 1
    
    # Additional LLAMA3-specific parameters
    use_flash_attention: bool = True
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
EOF

# Create training_config.py
cat > nemo_qa/config/training_config.py << 'EOF'
"""Training configuration for NeMo QA Chatbot."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Basic training parameters
    precision: int = 16
    devices: int = 1
    max_epochs: int = 3
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 8
    val_check_interval: float = 0.25
    
    # Learning rate parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Batch size and dataloader parameters
    batch_size: int = 4
    num_workers: int = 4
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 2
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
EOF

# Create lora_config.py
cat > nemo_qa/config/lora_config.py << 'EOF'
"""LoRA configuration for NeMo QA Chatbot."""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    
    # LoRA parameters
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = None
    bias: str = "none"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
EOF

# Create nim_config.py
cat > nemo_qa/config/nim_config.py << 'EOF'
"""NIM configuration for NeMo QA Chatbot."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class NIMConfig:
    """Configuration for NeMo Inference Microservice."""
    
    # Model parameters
    model_path: str = "models/llama3-8b-lora"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    
    # Server parameters
    batch_size: int = 1
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    
    # Runtime parameters
    tensor_parallel_size: int = 1
    use_trt_llm: bool = False
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
EOF