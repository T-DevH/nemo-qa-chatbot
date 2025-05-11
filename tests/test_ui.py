"""Tests for UI module."""

import os
import pytest
import gradio as gr
from nemo_qa.ui.app import create_ui
from nemo_qa.ui.components.chat import create_chat_component
from nemo_qa.ui.components.explainability import create_explainability_component
from nemo_qa.ui.components.visualization import create_attention_heatmap, create_confidence_visualization

@pytest.mark.skip(reason="Requires UI")
def test_create_ui():
    """Test create UI."""
    ui = create_ui("http://localhost:8000")
    
    assert ui is not None
    assert isinstance(ui, gr.Blocks)

@pytest.mark.skip(reason="Requires UI")
def test_create_chat_component():
    """Test create chat component."""
    chat_func = lambda x, y, z: ("Response", 0.8, [])
    
    component, chatbot, question, submit_btn = create_chat_component(chat_func)
    
    assert component is not None
    assert chatbot is not None
    assert question is not None
    assert submit_btn is not None

@pytest.mark.skip(reason="Requires UI")
def test_create_explainability_component():
    """Test create explainability component."""
    component, confidence, attention_heatmap = create_explainability_component()
    
    assert component is not None
    assert confidence is not None
    assert attention_heatmap is not None

def test_create_attention_heatmap():
    """Test create attention heatmap."""
    # Empty attention scores
    heatmap = create_attention_heatmap([])
    assert heatmap.shape == (10, 10)
    
    # Non-empty attention scores
    heatmap = create_attention_heatmap([0.1, 0.2, 0.3])
    assert heatmap.shape == (10, 10)

def test_create_confidence_visualization():
    """Test create confidence visualization."""
    # High confidence
    viz = create_confidence_visualization(0.9)
    assert viz["label"] == "High"
    assert viz["confidences"]["High"] == 0.9
    
    # Medium confidence
    viz = create_confidence_visualization(0.7)
    assert viz["label"] == "Medium"
    assert viz["confidences"]["Medium"] == 0.7
    
    # Low confidence
    viz = create_confidence_visualization(0.3)
    assert viz["label"] == "Low"
    assert viz["confidences"]["Low"] == 0.3
