cat > nim/Dockerfile << 'EOF'
FROM nvcr.io/nvidia/nemo:24.01

WORKDIR /app

# Install additional dependencies
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy NIM implementation
COPY nim/ /app/nim/
COPY nemo_qa/ /app/nemo_qa/

# Copy model artifacts (will be mounted in production)
COPY models/ /app/models/

# Set environment variables
ENV MODEL_PATH=/app/models/llama3-8b-lora
ENV MAX_LENGTH=512
ENV TEMPERATURE=0.7
ENV TOP_P=0.9
ENV TOP_K=50

# Expose port for NIM
EXPOSE 8000

# Run the NIM server
CMD ["python", "-m", "nim.app"]
EOF