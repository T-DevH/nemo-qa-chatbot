"""Data curation recipe for NeMo QA Chatbot."""

import os
import logging
from typing import Dict, Any, Optional
from nemo_qa.curator.document_processor import process_documents
from nemo_qa.curator.qa_generator import generate_qa_pairs
from nemo_qa.curator.quality_filters import filter_qa_pairs

logger = logging.getLogger(__name__)

def curator_recipe(
    input_dir: str,
    output_dir: str,
    model_path: str,
    document_filters: Optional[Dict[str, Any]] = None,
    qa_generation_config: Optional[Dict[str, Any]] = None,
    quality_filters_config: Optional[Dict[str, Any]] = None
):
    """Data curation recipe.
    
    Args:
        input_dir: Directory containing input documents
        output_dir: Directory to save processed data
        model_path: Path to the pretrained model
        document_filters: Document filtering configuration
        qa_generation_config: Q&A generation configuration
        quality_filters_config: Quality filtering configuration
    
    Returns:
        Statistics about the curation process
    """
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    processed_dir = os.path.join(output_dir, "processed")
    qa_dir = os.path.join(output_dir, "qa_pairs")
    filtered_dir = os.path.join(output_dir, "filtered")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(qa_dir, exist_ok=True)
    os.makedirs(filtered_dir, exist_ok=True)
    
    # Default configurations
    if document_filters is None:
        document_filters = {}
    
    if qa_generation_config is None:
        qa_generation_config = {
            "question_template": "Generate a question based on this text: {text}",
            "answer_template": "Answer the following question based on this text: {text}\nQuestion: {question}\nAnswer:",
            "batch_size": 4,
            "max_length": 512
        }
    
    if quality_filters_config is None:
        quality_filters_config = {
            "min_question_length": 10,
            "max_question_length": 200,
            "min_answer_length": 50,
            "max_answer_length": 1000,
            "min_relevance_score": 0.3,
            "diversity_clusters": 10
        }
    
    # Process documents
    logger.info("Processing documents...")
    doc_stats = process_documents(
        input_dir=input_dir,
        output_dir=processed_dir,
        filters=document_filters.get("filters", []),
        num_workers=document_filters.get("num_workers", 4),
        batch_size=document_filters.get("batch_size", 32)
    )
    
    # Generate Q&A pairs
    logger.info("Generating Q&A pairs...")
    qa_pairs = generate_qa_pairs(
        model_path=model_path,
        input_dir=processed_dir,
        output_dir=qa_dir,
        question_template=qa_generation_config.get("question_template"),
        answer_template=qa_generation_config.get("answer_template"),
        batch_size=qa_generation_config.get("batch_size"),
        max_length=qa_generation_config.get("max_length")
    )
    
    # Filter Q&A pairs
    logger.info("Filtering Q&A pairs...")
    filtered_pairs = filter_qa_pairs(
        qa_pairs=qa_pairs,
        output_dir=filtered_dir,
        min_question_length=quality_filters_config.get("min_question_length"),
        max_question_length=quality_filters_config.get("max_question_length"),
        min_answer_length=quality_filters_config.get("min_answer_length"),
        max_answer_length=quality_filters_config.get("max_answer_length"),
        min_relevance_score=quality_filters_config.get("min_relevance_score"),
        diversity_clusters=quality_filters_config.get("diversity_clusters")
    )
    
    # Create dataset splits
    from sklearn.model_selection import train_test_split
    
    train_pairs, temp_pairs = train_test_split(
        filtered_pairs, test_size=0.2, random_state=42
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.5, random_state=42
    )
    
    # Save dataset splits
    import json
    
    datasets_dir = os.path.join(output_dir, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    with open(os.path.join(datasets_dir, "train.jsonl"), "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    
    with open(os.path.join(datasets_dir, "val.jsonl"), "w") as f:
        for pair in val_pairs:
            f.write(json.dumps(pair) + "\n")
    
    with open(os.path.join(datasets_dir, "test.jsonl"), "w") as f:
        for pair in test_pairs:
            f.write(json.dumps(pair) + "\n")
    
    logger.info(f"Created dataset splits: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    # Return statistics
    return {
        "documents": doc_stats,
        "qa_pairs": len(qa_pairs),
        "filtered_pairs": len(filtered_pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs)
    }
