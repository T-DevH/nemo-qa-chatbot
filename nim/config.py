cat > nim/config.py << 'EOF'
"""NIM configuration."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class NIMConfig:
    """Configuration for NeMo Inference Microservice."""
    
    # Model parameters
    model_path: str = os.environ.get("MODEL_PATH", "models/llama3-8b-lora")
    max_length: int = int(os.environ.get("MAX_LENGTH", 512))
    temperature: float = float(os.environ.get("TEMPERATURE", 0.7))
    top_p: float = float(os.environ.get("TOP_P", 0.9))
    top_k: int = int(os.environ.get("TOP_K", 50))
    
    # Server parameters
    batch_size: int = int(os.environ.get("BATCH_SIZE", 1))
    max_batch_size: int = int(os.environ.get("MAX_BATCH_SIZE", 8))
    
    def to_dict(self):
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load NIM configuration.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    import json
    
    # Load from file if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Use default configuration
        config = NIMConfig().to_dict()
    
    return config
EOF