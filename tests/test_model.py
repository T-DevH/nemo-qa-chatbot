"""Tests for model module."""

import os
import pytest
import torch
from nemo_qa.modeling.model import load_model
from nemo_qa.modeling.lora import create_lora_config, enable_lora_for_model
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

@pytest.mark.skip(reason="Requires model")
def test_load_model():
    """Test load model."""
    model = load_model("models/base/llama3-8b")
    
    assert model is not None
    assert isinstance(model.model, MegatronGPTModel)

@pytest.mark.skip(reason="Requires model")
def test_generate():
    """Test generate."""
    model = load_model("models/base/llama3-8b")
    
    prompt = "Human: What is a test?\nAssistant:"
    output = model.generate(prompt)
    
    assert output is not None
    assert "text" in output
    assert isinstance(output["text"], list)
    assert len(output["text"]) > 0

@pytest.mark.skip(reason="Requires model")
def test_lora_config():
    """Test LoRA configuration."""
    lora_config = create_lora_config(
        r=16,
        alpha=32,
        dropout=0.05
    )
    
    assert lora_config["r"] == 16
    assert lora_config["alpha"] == 32
    assert lora_config["dropout"] == 0.05
    assert "q_proj" in lora_config["target_modules"]

@pytest.mark.skip(reason="Requires model")
def test_enable_lora():
    """Test enable LoRA."""
    model = MegatronGPTModel.from_pretrained("nvidia/nemo-llama3-8b")
    
    lora_config = create_lora_config()
    model_with_lora = enable_lora_for_model(model, lora_config)
    
    assert model_with_lora is not None
