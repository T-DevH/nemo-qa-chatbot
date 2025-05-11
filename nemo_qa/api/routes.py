# This file is intended for API routes.
# Add your route definitions here.

"""API routes for NeMo QA Chatbot."""

import logging
import torch
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

class Query(BaseModel):
    """Query model."""
    
    question: str
    context: str = ""
    history: List[List[str]] = []

class Response(BaseModel):
    """Response model."""
    
    response: str
    explainability: Optional[Dict[str, Any]] = None

def get_model(request: Request):
    """Get model from app state."""
    return request.app.state.model

@router.post("/chat", response_model=Response)
async def chat(query: Query, model=Depends(get_model)):
    """Chat endpoint.
    
    Args:
        query: Query model
        model: NeMo model
    
    Returns:
        Response model
    """
    # Format prompt
    prompt = format_prompt(query.question, query.context, query.history)
    
    # Generate response
    with torch.inference_mode():
        output = model.generate(
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            return_attention=True
        )
    
    # Extract response
    response = output["text"][0].replace(prompt, "").strip()
    
    # Process attention weights for explainability
    attention = output.get("attention_weights")
    explainability = process_attention(attention) if attention is not None else None
    
    return Response(
        response=response,
        explainability=explainability
    )

def format_prompt(question: str, context: str = "", history: List = None) -> str:
    """Format prompt for model input.
    
    Args:
        question: User question
        context: Optional context
        history: Conversation history
    
    Returns:
        Formatted prompt
    """
    if history is None:
        history = []
    
    # Format history
    formatted_history = ""
    for entry in history:
        formatted_history += f"Human: {entry[0]}\nAssistant: {entry[1]}\n\n"
    
    # Add context if provided
    context_str = f"Context: {context}\n\n" if context else ""
    
    # Format prompt
    prompt = f"{formatted_history}{context_str}Human: {question}\nAssistant:"
    
    return prompt

def process_attention(attention):
    """Process attention weights for explainability.
    
    Args:
        attention: Attention weights
    
    Returns:
        Explainability data
    """
    if attention is None:
        return None
    
    try:
        # Simplify attention for visualization
        # This is a placeholder implementation
        mean_attention = attention.mean(dim=0).tolist()
        
        return {
            "attention_scores": mean_attention
        }
    except Exception as e:
        logger.error(f"Error processing attention weights: {e}")
        return None