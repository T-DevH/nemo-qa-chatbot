# This file is intended for the API application.
# Add your API code here.

"""FastAPI application for NeMo QA Chatbot."""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from nemo_qa.api.routes import router
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

logger = logging.getLogger(__name__)

def create_app(model_path: str) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_path: Path to the model
    
    Returns:
        FastAPI application
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = MegatronGPTModel.restore_from(model_path)
    model.eval()
    
    # Create FastAPI app
    app = FastAPI(
        title="NeMo QA Chatbot API",
        description="API for NeMo QA Chatbot",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Add model to app state
    app.state.model = model
    
    # Include router
    app.include_router(router)
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "NeMo QA Chatbot API"}
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "ok"}
    
    return app