"""Tests for API module."""

import os
import pytest
from fastapi.testclient import TestClient
from nemo_qa.api.app import create_app
from nemo_qa.api.routes import format_prompt

@pytest.mark.skip(reason="Requires model")
def test_app():
    """Test FastAPI application."""
    app = create_app("models/base/llama3-8b")
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.skip(reason="Requires model")
def test_chat_endpoint():
    """Test chat endpoint."""
    app = create_app("models/base/llama3-8b")
    
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "question": "What is a test?",
            "context": "",
            "history": []
        }
    )
    
    assert response.status_code == 200
    assert "response" in response.json()

def test_format_prompt():
    """Test format prompt."""
    # Simple prompt
    prompt = format_prompt("What is a test?")
    assert prompt == "Human: What is a test?\nAssistant:"
    
    # Prompt with context
    prompt = format_prompt("What is a test?", "A test is a procedure.")
    assert prompt == "Context: A test is a procedure.\n\nHuman: What is a test?\nAssistant:"
    
    # Prompt with history
    prompt = format_prompt(
        "What is a test?",
        history=[["Hello", "Hi there"]]
    )
    assert prompt == "Human: Hello\nAssistant: Hi there\n\nHuman: What is a test?\nAssistant:"
    
    # Prompt with history and context
    prompt = format_prompt(
        "What is a test?",
        "A test is a procedure.",
        [["Hello", "Hi there"]]
    )
    assert prompt == "Human: Hello\nAssistant: Hi there\n\nContext: A test is a procedure.\n\nHuman: What is a test?\nAssistant:"
