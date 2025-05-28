# NeMo QA Chatbot
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA NeMo](https://img.shields.io/badge/NVIDIA-NeMo%202.0-green.svg)](https://github.com/NVIDIA/NeMo)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-blue.svg)](https://developer.nvidia.com/cuda-12-8-0-download-archive)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

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

### Prerequisites

Before starting, ensure you have:
1. NVIDIA GPU with CUDA 12.8 support
2. Docker and NVIDIA Container Toolkit installed
3. Sufficient disk space (at least 20GB)
4. NGC account and API token

### Container-based Setup

The project uses NVIDIA's NeMo container for a consistent development environment. The setup is managed through a Makefile for simplicity.

1. **Clone the repository**:
```bash
git clone https://github.com/T-DevH/nemo-qa-chatbot.git
cd nemo-qa-chatbot
```

2. **Build and Run the Container**:
```bash
# Build and start the container
make run

# To access the container shell later
make shell
```

The Makefile handles:
- Pulling the NVIDIA NeMo container (nvcr.io/nvidia/nemo:24.09)
- Setting up the development environment
- Installing system dependencies
- Installing Poetry for dependency management
- Installing PyTorch with CUDA 12.8 support
- Installing project dependencies
- Installing mamba-ssm for efficient training

### Manual Setup (Alternative)

If you prefer to set up without containers:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Poetry
pip install poetry

# Install dependencies
poetry install

# Install PyTorch with CUDA support
poetry run pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Install mamba-ssm
poetry run pip install mamba-ssm==2.2.2
```

## Usage

### Prerequisites

Before starting, ensure you have:
1. NVIDIA GPU with CUDA 12.8 support
2. Docker and NVIDIA Container Toolkit installed
3. Sufficient disk space (at least 20GB)
4. NGC account and API token

### 1. Install NGC CLI (if not already installed)

```bash
# Download NGC CLI
wget https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip ngccli_linux.zip

# Move to a system directory
sudo mv ngc /usr/local/bin/

# Verify installation
ngc --version
```

### 2. Configure NGC CLI

```bash
# Login to NGC
ngc config set

# Enter your NGC API key when prompted
# You can find your API key at: https://ngc.nvidia.com/setup/api-key
```

### 3. Download Base Model

```bash
# Basic download
python scripts/download_model.py

# With specific options
python scripts/download_model.py \
  --model_name llama3-8b \
  --output_dir models/base

# Force redownload if model exists
python scripts/download_model.py --force
```

The download script will:
- Verify sufficient disk space
- Check NGC CLI availability
- Download the model using NGC CLI
- Show download progress in real-time
- Handle any download errors gracefully

Available models:
- `llama3-8b`: LLAMA3 8B model
- `llama3.1-8b`: LLAMA3.1 8B model

### 4. Verify Model Download

After downloading, verify that the model files are present and correctly structured:

```bash
# Check the model directory structure
ls -la models/base/llama3-8b/llama-3_1-8b-nemo_v1.0

# Expected output should show:
# - llama3_1_8b.nemo (main model file, ~16GB)
# - Other configuration files
```

### 5. Curate Training Data

```bash
python scripts/curate_data.py --input_dir data/raw --output_dir data/processed
```

### 6. Fine-tune with LoRA

```bash
python scripts/train.py \
  --model_path models/base/llama3-8b \
  --train_data data/datasets/train.jsonl \
  --val_data data/datasets/val.jsonl \
  --output_dir models/finetuned
```

### 7. Evaluate the Model

```bash
python scripts/evaluate.py \
  --model_path models/finetuned/final_model \
  --test_data data/datasets/test.jsonl \
  --output_path evaluation_results.json
```

### 8. Deploy Chatbot

```bash
python scripts/deploy.py \
  --model_path models/finetuned/final_model
```

### 9. Export as NIM (Optional)

```bash
python scripts/export_nim.py \
  --model_path models/finetuned/final_model \
  --output_dir nim/export
```

### 10. Build and Run NIM Container (Optional)

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

## Citation
if you use this project in your reaserch or work, please consider citing: 
@misc{hammadou2025nemoqachatbot,
  author = {Hammadou, Tarik},
  title = {NeMo QA Chatbot: Production-Ready Q&A with LLAMA3 and NeMo 2.0},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/T-DevH/nemo-qa-chatbot}}
}

## Acknowledgments

- NVIDIA NeMo team for the amazing framework
- LLAMA3 for the base model
- All contributors who have helped shape this project 

## Development Setup with Docker

We use Docker for development to ensure a consistent environment across all developers and to leverage NVIDIA's pre-built NeMo container. This approach has several advantages:

1. **Pre-built Dependencies**: The NVIDIA NeMo container (`nvcr.io/nvidia/nemo:25.02`) comes with:
   - CUDA and cuDNN pre-installed and configured
   - PyTorch with CUDA support
   - Pre-built `transformer-engine` optimized for the container's environment
   - Other NVIDIA-specific optimizations

2. **Isolated Environment**: Docker provides an isolated environment that:
   - Prevents conflicts with system-level dependencies
   - Ensures consistent behavior across different machines
   - Makes it easy to switch between different versions of dependencies

3. **Simplified Setup**: New developers can start working with just:
   ```bash
   make run
   ```

### What Happens When You Run `make run`

The `make run` command orchestrates the following process:

1. **Container Pull**: Downloads the NVIDIA NeMo base image (`nvcr.io/nvidia/nemo:25.02`)

2. **Volume Mounting**: 
   - Your local project directory is mounted at `/workspace/nemo_qa_chatbot`
   - Changes made locally are immediately reflected in the container
   - Your code runs in the container but is edited on your host machine

3. **Dependency Installation**:
   - Installs Poetry inside the container
   - Runs `poetry install --without transformer-engine`
   - This installs all dependencies from `pyproject.toml` except `transformer-engine`
   - The pre-built `transformer-engine` from the container is used instead

### Key Files and Their Roles

| File                 | Used? | Role                                      |
|----------------------|-------|-------------------------------------------|
| `pyproject.toml`     | ✅    | Defines all your app's dependencies       |
| `Makefile`          | ✅    | Orchestrates container setup and launch   |
| `nemo-toolkit`      | ✅    | Installed from TOML via Poetry            |
| `transformer-engine` | ❌    | Pre-built in container (skipped in TOML)  |
| Your code/scripts    | ✅    | Mounted live from your host machine       |

### Why We Skip `transformer-engine` in Poetry

We intentionally skip installing `transformer-engine` via Poetry because:
1. The NVIDIA container already includes a pre-built, optimized version
2. Building from source can be complex and error-prone
3. The pre-built version is guaranteed to work with the container's CUDA setup

### Development Workflow

1. Start the container:
   ```bash
   make run
   ```

2. Access the container shell (in a new terminal):
   ```bash
   make shell
   ```

3. Your code is mounted at `/workspace/nemo_qa_chatbot`
   - Edit files on your host machine
   - Run code inside the container
   - Changes are immediately reflected

4. When done, simply exit the container:
   ```bash
   exit
   ``` 