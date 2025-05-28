FROM nvcr.io/nvidia/nemo:25.02

WORKDIR /workspace/nemo_qa_chatbot

# Install system dependencies
RUN apt update && \
    apt install -y \
    git \
    software-properties-common \
    build-essential \
    python3-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    gfortran && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.10 python3.10-dev python3.10-venv

# Install poetry
RUN pip install poetry

# Copy project files
COPY . .

# Set up poetry environment and install dependencies
RUN poetry env use python3.10 && \
    poetry run pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    poetry install --no-root

# Set environment variables
ENV PYTHONPATH=/workspace/nemo_qa_chatbot

# Default command
CMD ["bash"] 