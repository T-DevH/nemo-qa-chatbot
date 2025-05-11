# NeMo QA Chatbot

A production-ready Q&A chatbot built with NVIDIA NeMo 2.0 and LLAMA3, featuring high-quality data curation and efficient LoRA fine-tuning.

## Overview

This project implements a comprehensive Question-Answering chatbot using NVIDIA's NeMo 2.0 framework with LLAMA3 8B and Low-Rank Adaptation (LoRA) fine-tuning. It addresses the critical importance of high-quality training data through a robust curation pipeline and provides explainability features to enhance user trust.

This is the second part of a 6-part workshop series on practical LLM implementation, building on the foundation established in [Part 1: Practical Guide to Fine-Tuning LLMs with NVIDIA NeMo and LoRA](https://medium.com/@thammadou/practical-guide-to-fine-tuning-llms-with-nvidia-nemo-and-lora-4af8ddc030ff).

## Key Features

- **Data Quality Pipeline**: Comprehensive data curation with document processing, QA generation, and multi-dimensional quality filtering
- **Efficient LoRA Fine-tuning**: Parameter-efficient adaptation of LLAMA3 8B with optimized hyperparameters
- **Explainable AI**: Interface with attention visualization and confidence metrics
- **Production-Ready Deployment**: FastAPI backend, Gradio UI, and NeMo Inference Microservice integration
- **Comprehensive Testing**: Test suite covering all modules for quality assurance

## Architecture

The project follows a modular design with clear separation of concerns:

- **Configuration**: Python-based configuration (not YAML) as per NeMo 2.0 recommendations
- **Data Curation**: Document processing, QA generation, and quality filtering
- **Modeling**: Model loading, LoRA integration, and evaluation metrics
- **Recipes**: Reusable workflows for data curation, training, and inference
- **API**: FastAPI-based REST API for serving the model
- **UI**: Gradio-based user interface with explainability features
- **NIM**: NeMo Inference Microservice for containerized deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/T-DevH/nemo-qa-chatbot.git
cd nemo-qa-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Usage

### 1. Download Base Model

```bash
python scripts/download_model.py
```

### 2. Curate Training Data

```bash
python scripts/curate_data.py --input_dir data/raw --output_dir data/processed
```

### 3. Fine-tune with LoRA

```bash
python scripts/train.py \
  --model_path models/base/llama3-8b \
  --train_data data/datasets/train.jsonl \
  --val_data data/datasets/val.jsonl \
  --output_dir models/finetuned
```

### 4. Evaluate the Model

```bash
python scripts/evaluate.py \
  --model_path models/finetuned/final_model \
  --test_data data/datasets/test.jsonl \
  --output_path evaluation_results.json
```

### 5. Deploy Chatbot

```bash
python scripts/deploy.py \
  --model_path models/finetuned/final_model
```

### 6. Export as NIM (Optional)

```bash
python scripts/export_nim.py \
  --model_path models/finetuned/final_model \
  --output_dir nim/export
```

### 7. Build and Run NIM Container (Optional)

```bash
cd nim/export
docker build -t llama3-qa-chatbot-nim .
docker run -p 8000:8000 --gpus all llama3-qa-chatbot-nim
```

## Jupyter Notebooks

The project includes several Jupyter notebooks for exploration and demonstration:

1. **Data Exploration**: Explore the data curation process
2. **Model Analysis**: Analyze model outputs and performance
3. **Interactive Demo**: Try out the chatbot in an interactive environment

## Project Structure

```
nemo-qa-chatbot/
├── nemo_qa/               # Main package
│   ├── config/            # Python-based configuration
│   ├── curator/           # Data curation module
│   ├── recipes/           # NeMo 2.0 recipes
│   ├── modeling/          # Model implementation
│   ├── api/               # FastAPI implementation
│   └── ui/                # Gradio UI implementation
├── nim/                   # NeMo Inference Microservice
├── scripts/               # Command-line scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
└── data/                  # Data directory
```

## Coming Soon in the Workshop Series

This project is part of a 6-part workshop series:

1. ✅ **Foundation**: Fine-tuning basics with NVIDIA NeMo and LoRA
2. ✅ **Quality**: Data curation and optimized fine-tuning (this project)
3. 🔜 **Advanced Reasoning**: Chain-of-thought implementation
4. 🔜 **Alignment**: RLHF and alignment techniques
5. 🔜 **Multimodal**: Multi-modal capabilities and RAG
6. 🔜 **Deployment**: Enterprise deployment and monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA NeMo team for the amazing framework
- LLAMA3 for the base model
- All contributors who have helped shape this project 