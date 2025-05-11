cat > nim/app.py << 'EOF'
"""NIM application entry point."""

import os
import logging
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nim.model_handler import LLAMA3QAChatbotHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load configuration
config_path = os.environ.get("NIM_CONFIG_PATH", "nim_config.json")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
else:
    config = {
        "model_path": os.environ.get("MODEL_PATH", "models/llama3-8b-lora"),
        "max_length": int(os.environ.get("MAX_LENGTH", 512)),
        "temperature": float(os.environ.get("TEMPERATURE", 0.7)),
        "top_p": float(os.environ.get("TOP_P", 0.9)),
        "top_k": int(os.environ.get("TOP_K", 50))
    }

# Initialize model handler
model_handler = LLAMA3QAChatbotHandler(**config)
model_handler.initialize(config["model_path"])

# Create FastAPI app
app = FastAPI(title="LLAMA3 Q&A Chatbot NIM")

@app.post("/v1/predict")
async def predict(request: Request):
    """Prediction endpoint."""
    request_data = await request.json()
    
    # Preprocess request
    model_input = model_handler.preprocess(request_data)
    
    # Run inference
    inference_output = model_handler.inference(model_input)
    
    # Postprocess output
    response = model_handler.postprocess(inference_output)
    
    return JSONResponse(content=response)

@app.get("/v1/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/v1/metadata")
async def metadata():
    """Metadata endpoint."""
    return {
        "name": "LLAMA3 Q&A Chatbot",
        "version": "0.1.0",
        "framework": "NeMo",
        "model": {
            "name": "llama3-8b",
            "precision": "fp16"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF