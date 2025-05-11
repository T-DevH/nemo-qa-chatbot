"""Quality filtering for Q&A pairs."""

import os
import logging
import json
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class QualityFilters:
    """Quality filters for Q&A pairs."""
    
    @staticmethod
    def length_filter(
        qa_pairs: List[Dict[str, Any]],
        min_question_length: int = 10,
        max_question_length: int = 200,
        min_answer_length: int = 50,
        max_answer_length: int = 1000
    ) -> List[Dict[str, Any]]:
        """Filter Q&A pairs by length.
        
        Args:
            qa_pairs: List of Q&A pairs
            min_question_length: Minimum question length
            max_question_length: Maximum question length
            min_answer_length: Minimum answer length
            max_answer_length: Maximum answer length
        
        Returns:
            Filtered Q&A pairs
        """
        filtered_pairs = []
        
        for pair in qa_pairs:
            question = pair["question"]
            answer = pair["answer"]
            
            if (len(question) >= min_question_length and 
                len(question) <= max_question_length and
                len(answer) >= min_answer_length and
                len(answer) <= max_answer_length):
                filtered_pairs.append(pair)
        
        logger.info(f"Length filter: {len(qa_pairs)} -> {len(filtered_pairs)}")
        
        return filtered_pairs
    
    @staticmethod
    def relevance_filter(
        qa_pairs: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Filter Q&A pairs by relevance.
        
        Args:
            qa_pairs: List of Q&A pairs
            min_score: Minimum relevance score
        
        Returns:
            Filtered Q&A pairs
        """
        filtered_pairs = []
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        
        for pair in qa_pairs:
            question = pair["question"]
            answer = pair["answer"]
            
            # Calculate relevance score
            try:
                vectors = vectorizer.fit_transform([question, answer])
                similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                
                # Add relevance score to pair
                pair["relevance_score"] = float(similarity)
                
                if similarity >= min_score:
                    filtered_pairs.append(pair)
            except:
                # Skip pairs that cause errors
                continue
        
        logger.info(f"Relevance filter: {len(qa_pairs)} -> {len(filtered_pairs)}")
        
        return filtered_pairs
    
    @staticmethod
    def diversity_filter(
        qa_pairs: List[Dict[str, Any]],
        min_clusters: int = 10
    ) -> List[Dict[str, Any]]:
        """Filter Q&A pairs by diversity.
        
        Args:
            qa_pairs: List of Q&A pairs
            min_clusters: Minimum number of clusters
        
        Returns:
            Filtered Q&A pairs
        """
        # If fewer pairs than clusters, return all pairs
        if len(qa_pairs) <= min_clusters:
            return qa_pairs
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words="english")
        
        # Extract questions
        questions = [pair["question"] for pair in qa_pairs]
        
        # Calculate TF-IDF vectors
        try:
            vectors = vectorizer.fit_transform(questions)
            
            # Perform K-means clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min_clusters, random_state=42)
            clusters = kmeans.fit_predict(vectors)
            
            # Select representative from each cluster
            filtered_pairs = []
            for i in range(min_clusters):
                cluster_indices = np.where(clusters == i)[0]
                
                if len(cluster_indices) > 0:
                    # Choose pair closest to centroid
                    centroid = kmeans.cluster_centers_[i:i+1]
                    similarities = cosine_similarity(vectors[cluster_indices], centroid)
                    best_idx = cluster_indices[np.argmax(similarities)]
                    filtered_pairs.append(qa_pairs[best_idx])
            
            logger.info(f"Diversity filter: {len(qa_pairs)} -> {len(filtered_pairs)}")
            
            return filtered_pairs
        except:
            # If clustering fails, return original pairs
            logger.warning("Diversity filtering failed, returning original pairs")
            return qa_pairs

def filter_qa_pairs(
    qa_pairs: List[Dict[str, Any]],
    output_dir: str,
    min_question_length: int = 10,
    max_question_length: int = 200,
    min_answer_length: int = 50,
    max_answer_length: int = 1000,
    min_relevance_score: float = 0.3,
    diversity_clusters: int = 10
):
    """Filter Q&A pairs by quality.
    
    Args:
        qa_pairs: List of Q&A pairs
        output_dir: Directory to save filtered Q&A pairs
        min_question_length: Minimum question length
        max_question_length: Maximum question length
        min_answer_length: Minimum answer length
        max_answer_length: Maximum answer length
        min_relevance_score: Minimum relevance score
        diversity_clusters: Number of diversity clusters
    
    Returns:
        Filtered Q&A pairs
    """
    logger.info(f"Filtering {len(qa_pairs)} Q&A pairs")
    
    # Apply filters
    filtered_pairs = QualityFilters.length_filter(
        qa_pairs,
        min_question_length=min_question_length,
        max_question_length=max_question_length,
        min_answer_length=min_answer_length,
        max_answer_length=max_answer_length
    )
    
    filtered_pairs = QualityFilters.relevance_filter(
        filtered_pairs,
        min_score=min_relevance_score
    )
    
    filtered_pairs = QualityFilters.diversity_filter(
        filtered_pairs,
        min_clusters=diversity_clusters
    )
    
    # Save filtered Q&A pairs
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "filtered_qa_pairs.jsonl")
    
    with open(output_file, "w") as f:
        for pair in filtered_pairs:
            f.write(json.dumps(pair) + "\n")
    
    logger.info(f"Filtered Q&A pairs: {len(qa_pairs)} -> {len(filtered_pairs)}")
    logger.info(f"Saved filtered Q&A pairs to {output_file}")
    
    return filtered_pairs
