"""Q&A pair generation for NeMo QA Chatbot."""

import os
import logging
import json
from typing import List, Dict, Any, Optional
import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

class QAGenerator:
    """Generate Q&A pairs from documents."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        batch_size: int = 4,
        max_length: int = 512,
        device: str = None
    ):
        """Initialize Q&A generator.
        
        Args:
            model_path: Path to the pretrained model
            output_dir: Directory to save generated Q&A pairs
            batch_size: Batch size for generation
            max_length: Maximum length of generated text
            device: Device to run the model on
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = MegatronGPTModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def generate_qa_pairs(
        self,
        documents: List[Dict[str, Any]],
        question_template: str = "Generate a question based on this text: {text}",
        answer_template: str = "Answer the following question based on this text: {text}\nQuestion: {question}\nAnswer:"
    ):
        """Generate Q&A pairs from documents.
        
        Args:
            documents: List of documents
            question_template: Template for generating questions
            answer_template: Template for generating answers
        
        Returns:
            List of Q&A pairs
        """
        logger.info(f"Generating Q&A pairs for {len(documents)} documents")
        qa_pairs = []
        
        for i, doc in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(documents)} documents")
            
            text = doc["text"]
            
            # Generate question
            question_prompt = question_template.format(text=text)
            with torch.inference_mode():
                question = self.model.generate(
                    question_prompt,
                    max_length=self.max_length // 4,
                    temperature=0.7
                )[0]
            
            # Generate answer
            answer_prompt = answer_template.format(text=text, question=question)
            with torch.inference_mode():
                answer = self.model.generate(
                    answer_prompt,
                    max_length=self.max_length,
                    temperature=0.3
                )[0]
            
            # Create Q&A pair
            qa_pair = {
                "document_id": doc.get("id", i),
                "text": text,
                "question": question,
                "answer": answer,
                "metadata": doc.get("metadata", {})
            }
            
            qa_pairs.append(qa_pair)
        
        # Save Q&A pairs
        output_file = os.path.join(self.output_dir, "qa_pairs.jsonl")
        with open(output_file, "w") as f:
            for pair in qa_pairs:
                f.write(json.dumps(pair) + "\n")
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        logger.info(f"Saved Q&A pairs to {output_file}")
        
        return qa_pairs

def generate_qa_pairs(
    model_path: str,
    input_dir: str,
    output_dir: str,
    question_template: str = "Generate a question based on this text: {text}",
    answer_template: str = "Answer the following question based on this text: {text}\nQuestion: {question}\nAnswer:",
    batch_size: int = 4,
    max_length: int = 512
):
    """Generate Q&A pairs from documents.
    
    Args:
        model_path: Path to the pretrained model
        input_dir: Directory containing processed documents
        output_dir: Directory to save generated Q&A pairs
        question_template: Template for generating questions
        answer_template: Template for generating answers
        batch_size: Batch size for generation
        max_length: Maximum length of generated text
    
    Returns:
        List of Q&A pairs
    """
    # Load documents
    documents = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "r") as f:
                if filename.endswith(".json"):
                    doc = json.load(f)
                    documents.append(doc)
                else:
                    for line in f:
                        doc = json.loads(line.strip())
                        documents.append(doc)
    
    # Generate Q&A pairs
    generator = QAGenerator(
        model_path=model_path,
        output_dir=output_dir,
        batch_size=batch_size,
        max_length=max_length
    )
    
    return generator.generate_qa_pairs(
        documents=documents,
        question_template=question_template,
        answer_template=answer_template
    )
