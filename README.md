# NeMo QA Chatbot

A production-ready Q&A chatbot built with NVIDIA NeMo 2.0 and LLAMA3, featuring high-quality data curation and efficient LoRA fine-tuning.

## Features

- Complete data curation pipeline using NeMo Curator
- Efficient LoRA fine-tuning with hyperparameter optimization
- Explainable AI features built into the chatbot interface
- Production-ready deployment with FastAPI and Gradio UI
- NeMo Inference Microservices (NIM) implementation

## Installation

```bash
# Clone the repository
git clone https://github.com/T-DevH/nemo-qa-chatbot.git
cd nemo-qa-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt