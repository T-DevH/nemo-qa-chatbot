# This file is intended for API initialization.
# Add your initialization code here.

"""API implementation for NeMo QA Chatbot.

This module provides API endpoints for the NeMo QA Chatbot using FastAPI.
"""

from nemo_qa.api.app import create_app
from nemo_qa.api.routes import router

__all__ = ["create_app", "router"]