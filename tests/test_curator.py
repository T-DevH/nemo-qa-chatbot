"""Tests for curator module."""

import os
import pytest
import tempfile
import json
from nemo_qa.curator.document_processor import process_documents
from nemo_qa.curator.qa_generator import generate_qa_pairs
from nemo_qa.curator.quality_filters import filter_qa_pairs

@pytest.mark.skip(reason="Requires model and documents")
def test_document_processor():
    """Test document processor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test documents
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        with open(os.path.join(input_dir, "test.json"), "w") as f:
            json.dump({
                "text": "This is a test document."
            }, f)
        
        # Process documents
        output_dir = os.path.join(temp_dir, "output")
        stats = process_documents(
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        # Check stats
        assert stats["input_files"] == 1
        assert stats["output_files"] == 1
        
        # Check output
        assert os.path.exists(os.path.join(output_dir, "test.json"))

@pytest.mark.skip(reason="Requires model and documents")
def test_qa_generator():
    """Test Q&A generator."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test documents
        input_dir = os.path.join(temp_dir, "input")
        os.makedirs(input_dir, exist_ok=True)
        
        with open(os.path.join(input_dir, "test.json"), "w") as f:
            json.dump({
                "text": "This is a test document."
            }, f)
        
        # Generate Q&A pairs
        output_dir = os.path.join(temp_dir, "output")
        qa_pairs = generate_qa_pairs(
            model_path="models/base/llama3-8b",
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        # Check output
        assert len(qa_pairs) > 0
        assert os.path.exists(os.path.join(output_dir, "qa_pairs.jsonl"))

@pytest.mark.skip(reason="Requires Q&A pairs")
def test_quality_filters():
    """Test quality filters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test Q&A pairs
        qa_pairs = [
            {
                "question": "What is the test document about?",
                "answer": "The test document is about testing."
            },
            {
                "question": "Is this a test?",
                "answer": "Yes, this is a test."
            }
        ]
        
        # Filter Q&A pairs
        output_dir = os.path.join(temp_dir, "output")
        filtered_pairs = filter_qa_pairs(
            qa_pairs=qa_pairs,
            output_dir=output_dir
        )
        
        # Check output
        assert len(filtered_pairs) > 0
        assert os.path.exists(os.path.join(output_dir, "filtered_qa_pairs.jsonl"))
