cat > nemo_qa/config/training_config.py << 'EOF'
"""Training configuration for NeMo QA Chatbot.

This module defines the configuration for training the NeMo QA Chatbot model,
including training parameters, optimization settings, and evaluation configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union

@dataclass
class TrainingConfig:
    """Configuration for training.
    
    This class defines the configuration parameters for training the NeMo QA Chatbot model,
    including training parameters, optimization settings, and evaluation configurations.
    
    Attributes:
        precision: Precision for training (16, 32, or "bf16").
        devices: Number of devices to use for training.
        max_epochs: Maximum number of epochs to train.
        gradient_clip_val: Maximum gradient norm.
        accumulate_grad_batches: Number of batches to accumulate gradients for.
        val_check_interval: Interval for validation.
        learning_rate: Learning rate for optimizer.
        weight_decay: Weight decay for optimizer.
        warmup_steps: Number of warmup steps for learning rate scheduler.
        batch_size: Batch size per GPU.
        num_workers: Number of workers for data loading.
        early_stopping: Whether to use early stopping.
        patience: Patience for early stopping.
        save_top_k: Number of best models to save.
        save_last: Whether to save the last model.
        monitor: Metric to monitor for early stopping and model saving.
        mode: Mode for early stopping and model saving (min or max).
        log_every_n_steps: Log every n steps.
        val_every_n_epochs: Validate every n epochs.
        deterministic: Whether to use deterministic training.
        seed: Random seed for reproducibility.
        optimizer: Optimizer configuration.
        scheduler: Scheduler configuration.
        callbacks: Callback configurations.
    """
    
    # Basic training parameters
    precision: Union[int, str] = 16
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
    
    # Early stopping and model saving
    early_stopping: bool = True
    patience: int = 2
    save_top_k: int = 2
    save_last: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
    
    # Logging
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    
    # Reproducibility
    deterministic: bool = False
    seed: int = 42
    
    # Advanced configurations
    optimizer: Dict[str, Any] = field(default_factory=lambda: {
        "type": "adamw",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8
        }
    })
    
    scheduler: Dict[str, Any] = field(default_factory=lambda: {
        "type": "cosine_with_warmup",
        "params": {
            "warmup_steps": 100,
            "max_steps": 1000,
            "min_lr": 1e-6
        }
    })
    
    callbacks: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default values that depend on other parameters."""
        # Set default callbacks if not provided
        if not self.callbacks:
            self.callbacks = [
                {
                    "type": "model_checkpoint",
                    "params": {
                        "dirpath": "checkpoints",
                        "filename": "{epoch}-{val_loss:.2f}",
                        "save_top_k": self.save_top_k,
                        "save_last": self.save_last,
                        "monitor": self.monitor,
                        "mode": self.mode,
                        "verbose": True
                    }
                },
                {
                    "type": "early_stopping",
                    "params": {
                        "monitor": self.monitor,
                        "patience": self.patience,
                        "mode": self.mode,
                        "verbose": True
                    }
                } if self.early_stopping else None,
                {
                    "type": "lr_monitor",
                    "params": {
                        "logging_interval": "step"
                    }
                }
            ]
            # Remove None values
            self.callbacks = [cb for cb in self.callbacks if cb is not None]
        
        # Update optimizer and scheduler with learning rate
        self.optimizer["params"]["lr"] = self.learning_rate
        self.optimizer["params"]["weight_decay"] = self.weight_decay
        
        # Update scheduler with warmup steps
        self.scheduler["params"]["warmup_steps"] = self.warmup_steps
    
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
    def load(cls, config_path: str) -> "TrainingConfig":
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
    
    def get_lightning_trainer_kwargs(self) -> Dict[str, Any]:
        """Get PyTorch Lightning Trainer kwargs.
        
        Returns:
            PyTorch Lightning Trainer kwargs.
        """
        return {
            "precision": self.precision,
            "devices": self.devices,
            "max_epochs": self.max_epochs,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "val_check_interval": self.val_check_interval,
            "log_every_n_steps": self.log_every_n_steps,
            "check_val_every_n_epoch": self.val_every_n_epochs,
            "deterministic": self.deterministic,
            "accelerator": "gpu" if self.devices > 0 else "cpu",
        }
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration.
        
        Returns:
            Optimizer configuration.
        """
        return self.optimizer
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration.
        
        Returns:
            Scheduler configuration.
        """
        return self.scheduler
    
    def get_callbacks(self) -> List[Dict[str, Any]]:
        """Get callback configurations.
        
        Returns:
            Callback configurations.
        """
        return self.callbacks

# Example usage
if __name__ == "__main__":
    # Create a default configuration
    config = TrainingConfig()
    
    # Print the configuration
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save the configuration
    config.save("training_config.json")
    
    # Load the configuration
    loaded_config = TrainingConfig.load("training_config.json")
    
    # Get PyTorch Lightning Trainer kwargs
    trainer_kwargs = config.get_lightning_trainer_kwargs()
EOF