cat > nemo_qa/recipes/lora_recipe.py << 'EOF'
"""LoRA fine-tuning recipe for NeMo QA Chatbot."""

import os
import logging
from typing import Dict, Any, Optional
import torch
import pytorch_lightning as pl
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPPlugin

logger = logging.getLogger(__name__)

def create_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[list] = None
) -> Dict[str, Any]:
    """Create LoRA configuration dictionary.
    
    Args:
        r: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: List of target modules
    
    Returns:
        LoRA configuration dictionary
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    return {
        "r": r,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": target_modules,
        "bias": "none",
        "modules_to_save": []
    }

def setup_lora_finetuning(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    lora_config: Optional[Dict[str, Any]] = None,
    training_args: Optional[Dict[str, Any]] = None
) -> pl.Trainer:
    """Set up LoRA fine-tuning using NeMo 2.0 approach.
    
    Args:
        model_path: Path to the pretrained model
        train_data_path: Path to the training data
        val_data_path: Path to the validation data
        output_dir: Directory to save the fine-tuned model
        lora_config: LoRA configuration
        training_args: Training arguments
    
    Returns:
        PyTorch Lightning trainer and model
    """
    # Create default configs if not provided
    if lora_config is None:
        lora_config = create_lora_config()
    
    if training_args is None:
        training_args = {
            "precision": 16,
            "devices": 1,
            "max_epochs": 3,
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 8,
            "val_check_interval": 0.25,
            "batch_size": 4,
            "num_workers": 4
        }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = MegatronGPTModel.from_pretrained(model_path)
    
    # Enable LoRA
    logger.info(f"Enabling LoRA with config: {lora_config}")
    model.enable_lora(**lora_config)
    
    # Set up training data
    logger.info(f"Setting up training data from {train_data_path}")
    model.setup_training_data(train_data_config={
        'file_path': train_data_path,
        'batch_size': training_args.get("batch_size", 4),
        'shuffle': True,
        'num_workers': training_args.get("num_workers", 4),
        'drop_last': True
    })
    
    # Set up validation data
    logger.info(f"Setting up validation data from {val_data_path}")
    model.setup_validation_data(val_data_config={
        'file_path': val_data_path,
        'batch_size': training_args.get("batch_size", 4),
        'shuffle': False,
        'num_workers': training_args.get("num_workers", 4),
        'drop_last': False
    })
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = pl.Trainer(
        precision=training_args.get("precision", 16),
        devices=training_args.get("devices", 1),
        accelerator="gpu",
        max_epochs=training_args.get("max_epochs", 3),
        gradient_clip_val=training_args.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_args.get("accumulate_grad_batches", 8),
        val_check_interval=training_args.get("val_check_interval", 0.25),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(output_dir, "checkpoints"),
                filename="{epoch}-{val_loss:.2f}",
                monitor="val_loss",
                save_top_k=2
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=training_args.get("patience", 2)
            )
        ],
        plugins=[NLPDDPPlugin()]
    )
    
    return trainer, model

def lora_finetuning_recipe(
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    lora_config: Optional[Dict[str, Any]] = None,
    training_args: Optional[Dict[str, Any]] = None
) -> str:
    """LoRA fine-tuning recipe.
    
    Args:
        model_path: Path to the pretrained model
        train_data_path: Path to the training data
        val_data_path: Path to the validation data
        output_dir: Directory to save the fine-tuned model
        lora_config: LoRA configuration
        training_args: Training arguments
    
    Returns:
        Path to the fine-tuned model
    """
    # Set up fine-tuning
    trainer, model = setup_lora_finetuning(
        model_path=model_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        lora_config=lora_config,
        training_args=training_args
    )
    
    # Train model
    logger.info("Starting training")
    trainer.fit(model)
    
    # Save the fine-tuned model
    model_save_path = os.path.join(output_dir, "final_model")
    logger.info(f"Saving fine-tuned model to {model_save_path}")
    model.save_to(model_save_path)
    
    return model_save_path
EOF