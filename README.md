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
```

## Usage

1. Start the FastAPI server:
```bash
python src/app.py
```

2. Access the Gradio UI at `http://localhost:7860`

3. Enter your questions in the chat interface and get AI-powered responses

## Configuration

The chatbot can be configured through environment variables or a config file:

- `MODEL_PATH`: Path to the fine-tuned model
- `MAX_LENGTH`: Maximum response length
- `TEMPERATURE`: Sampling temperature for responses

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA NeMo team for the amazing framework
- LLAMA3 for the base model
- All contributors who have helped shape this project 

## Project Under Developmenmt | Testing => Coming Soon 