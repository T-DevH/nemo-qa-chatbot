"""Document processing utilities for NeMo QA Chatbot."""

import os
import logging
from typing import List, Dict, Any, Optional
from nemo.collections.nlp.data.language_modeling.megatron.curator_dataloader import CuratorDataloader

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Process documents for Q&A generation."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        num_workers: int = 4,
        batch_size: int = 32
    ):
        """Initialize document processor.
        
        Args:
            input_dir: Directory containing input documents
            output_dir: Directory to save processed documents
            num_workers: Number of workers for parallel processing
            batch_size: Batch size for processing
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize curator
        self.curator = CuratorDataloader(
            input_dir=input_dir,
            output_dir=output_dir,
            num_workers=num_workers
        )
    
    def add_filters(self, filters: List[Dict[str, Any]]):
        """Add filters to the curator.
        
        Args:
            filters: List of filter configurations
        """
        for filter_config in filters:
            self.curator.add_filter(**filter_config)
    
    def process(self):
        """Process documents.
        
        Returns:
            Statistics about the processing
        """
        logger.info(f"Processing documents from {self.input_dir}")
        self.curator.process()
        logger.info(f"Documents processed and saved to {self.output_dir}")
        
        # Return statistics
        return {
            "input_files": len(os.listdir(self.input_dir)),
            "output_files": len(os.listdir(self.output_dir))
        }

def process_documents(
    input_dir: str,
    output_dir: str,
    filters: Optional[List[Dict[str, Any]]] = None,
    num_workers: int = 4,
    batch_size: int = 32
):
    """Process documents.
    
    Args:
        input_dir: Directory containing input documents
        output_dir: Directory to save processed documents
        filters: List of filter configurations
        num_workers: Number of workers for parallel processing
        batch_size: Batch size for processing
    
    Returns:
        Statistics about the processing
    """
    processor = DocumentProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        num_workers=num_workers,
        batch_size=batch_size
    )
    
    if filters:
        processor.add_filters(filters)
    
    return processor.process()
