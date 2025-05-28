.PHONY: build run shell

CONTAINER=nvcr.io/nvidia/nemo:24.09
PROJECT_DIR=$(shell pwd)
CONTAINER_NAME=nemo_qa_dev

build:
	docker pull --quiet $(CONTAINER)

run: build
	docker run --gpus all -it --rm \
	  --name $(CONTAINER_NAME) \
	  -v $(PROJECT_DIR):/workspace/nemo_qa_chatbot \
	  -w /workspace/nemo_qa_chatbot \
	  --shm-size=16g \
	  --ulimit memlock=-1 --ulimit stack=67108864 \
	  $(CONTAINER) \
	  bash -c "apt update && apt install -y git && pip install poetry && poetry install --no-root && poetry run pip install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && poetry run pip install mamba-ssm==2.2.2 && bash"

shell:
	docker exec -it $(CONTAINER_NAME) bash