#!/usr/bin/env python3
"""Deploy the Q&A chatbot."""

import os
import argparse
import logging
import torch
import threading
import uvicorn
from nemo_qa.recipes.inference_recipe import load_model_for_inference
from nemo_qa.api.app import create_app
from nemo_qa.ui.app import create_ui

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def deploy_api(model_path, host="0.0.0.0", port=8000):
    """Deploy the API server.
    
    Args:
        model_path: Path to the model
        host: Host to bind
        port: Port to bind
    """
    # Create FastAPI app
    app = create_app(model_path)
    
    # Run the server
    logger.info(f"Starting API server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)

def deploy_ui(api_url, host="0.0.0.0", port=7860):
    """Deploy the UI server.
    
    Args:
        api_url: URL of the API server
        host: Host to bind
        port: Port to bind
    """
    # Create Gradio app
    app = create_ui(api_url)
    
    # Run the server
    logger.info(f"Starting UI server on {host}:{port}...")
    app.launch(server_name=host, server_port=port)

def main():
    parser = argparse.ArgumentParser(description="Deploy the Q&A chatbot")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--api_only",
        action="store_true",
        help="Deploy API only (no UI)"
    )
    parser.add_argument(
        "--ui_only",
        action="store_true",
        help="Deploy UI only (no API)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind"
    )
    parser.add_argument(
        "--api_port",
        type=int,
        default=8000,
        help="API port"
    )
    parser.add_argument(
        "--ui_port",
        type=int,
        default=7860,
        help="UI port"
    )
    args = parser.parse_args()
    
    # Deploy API if requested
    if not args.ui_only:
        api_thread = threading.Thread(
            target=deploy_api,
            args=(args.model_path, args.host, args.api_port)
        )
        api_thread.start()
    
    # Deploy UI if requested
    if not args.api_only:
        ui_thread = threading.Thread(
            target=deploy_ui,
            args=(f"http://{args.host}:{args.api_port}", args.host, args.ui_port)
        )
        ui_thread.start()
    
    # Wait for threads to complete
    if not args.ui_only:
        api_thread.join()
    if not args.api_only:
        ui_thread.join()

if __name__ == "__main__":
    main()
