cat > nemo_qa/config/nim_config.py << 'EOF'
"""NIM configuration for NeMo QA Chatbot.

This module defines the configuration for NeMo Inference Microservices (NIM),
including model parameters, server settings, and runtime configurations.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

@dataclass
class NIMConfig:
    """Configuration for NeMo Inference Microservice.
    
    This class defines the configuration parameters for NeMo Inference Microservices (NIM),
    including model parameters, server settings, and runtime configurations.
    
    Attributes:
        model_path: Path to the model.
        max_length: Maximum length of generated text.
        temperature: Sampling temperature.
        top_p: Top-p sampling parameter.
        top_k: Top-k sampling parameter.
        batch_size: Batch size for inference.
        max_batch_size: Maximum batch size for inference.
        max_sequence_length: Maximum sequence length for input.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        use_trt_llm: Whether to use TensorRT-LLM for inference.
        trt_max_batch_size: Maximum batch size for TensorRT-LLM.
        trt_max_beam_width: Maximum beam width for TensorRT-LLM.
        trt_max_input_len: Maximum input length for TensorRT-LLM.
        trt_max_output_len: Maximum output length for TensorRT-LLM.
        trt_build_dir: Build directory for TensorRT-LLM.
        trt_engine_dir: Engine directory for TensorRT-LLM.
        host: Host to bind the server to.
        port: Port to bind the server to.
        log_level: Log level for the server.
        timeout: Timeout for requests in seconds.
        workers: Number of worker processes.
    """
    
    # Model parameters
    model_path: str = "models/llama3-8b-lora"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
    # Server parameters
    batch_size: int = 1
    max_batch_size: int = 8
    max_sequence_length: int = 2048
    
    # Runtime parameters
    tensor_parallel_size: int = 1
    use_trt_llm: bool = False
    
    # TensorRT-LLM parameters
    trt_max_batch_size: int = 8
    trt_max_beam_width: int = 1
    trt_max_input_len: int = 2048
    trt_max_output_len: int = 512
    trt_build_dir: str = "trt_build"
    trt_engine_dir: str = "trt_engines"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    timeout: int = 60
    workers: int = 1
    
    # Additional parameters
    enable_attention_visualization: bool = True
    enable_metrics: bool = True
    
    # Environment variables
    env_vars: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        return {k: v for k, v in self.__dict__.items()}
    
    def save(self, config_path: str) -> None:
        """Save configuration to a JSON file.
        
        Args:
            config_path: Path to save the configuration.
        """
        import json
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, config_path: str) -> "NIMConfig":
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Loaded configuration.
        """
        import json
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def get_docker_env_vars(self) -> Dict[str, str]:
        """Get environment variables for Docker container.
        
        Returns:
            Environment variables for Docker container.
        """
        env_vars = self.env_vars.copy()
        env_vars.update({
            "MODEL_PATH": self.model_path,
            "MAX_LENGTH": str(self.max_length),
            "TEMPERATURE": str(self.temperature),
            "TOP_P": str(self.top_p),
            "TOP_K": str(self.top_k),
            "BATCH_SIZE": str(self.batch_size),
            "MAX_BATCH_SIZE": str(self.max_batch_size),
            "MAX_SEQUENCE_LENGTH": str(self.max_sequence_length),
            "TENSOR_PARALLEL_SIZE": str(self.tensor_parallel_size),
            "USE_TRT_LLM": str(int(self.use_trt_llm)),
            "HOST": self.host,
            "PORT": str(self.port),
            "LOG_LEVEL": self.log_level,
            "TIMEOUT": str(self.timeout),
            "WORKERS": str(self.workers),
        })
        return env_vars
    
    def get_nim_manifest(self) -> Dict[str, Any]:
        """Get NIM manifest configuration.
        
        Returns:
            NIM manifest configuration.
        """
        return {
            "name": "llama3-qa-chatbot",
            "version": "0.1.0",
            "description": "Q&A chatbot built with NVIDIA NeMo 2.0 and LLAMA3",
            "author": "T-DevH",
            "license": "Apache-2.0",
            "repository": "https://github.com/T-DevH/nemo-qa-chatbot",
            "documentation": "https://github.com/T-DevH/nemo-qa-chatbot/blob/main/README.md",
            "tags": [
                "llm",
                "qa",
                "chatbot",
                "nemo",
                "llama3"
            ],
            "inputs": [
                {
                    "name": "question",
                    "description": "User question",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "context",
                    "description": "Optional context",
                    "type": "string",
                    "required": False
                },
                {
                    "name": "history",
                    "description": "Conversation history",
                    "type": "array",
                    "required": False
                }
            ],
            "outputs": [
                {
                    "name": "response",
                    "description": "Model response",
                    "type": "string"
                },
                {
                    "name": "explainability",
                    "description": "Explainability data",
                    "type": "object"
                }
            ],
            "runtime": {
                "nvidia_gpu": "required",
                "cpu_arch": "x86_64",
                "container": {
                    "ports": [
                        self.port
                    ],
                    "env": [
                        {
                            "name": key,
                            "description": f"Environment variable for {key}",
                            "default": value
                        }
                        for key, value in self.get_docker_env_vars().items()
                    ]
                }
            }
        }
    
    def get_nim_config_dict(self) -> Dict[str, Any]:
        """Get NIM configuration dictionary.
        
        Returns:
            NIM configuration dictionary.
        """
        return {
            "model": {
                "path": self.model_path,
                "parameters": {
                    "max_length": self.max_length,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "repetition_penalty": self.repetition_penalty,
                    "presence_penalty": self.presence_penalty,
                    "frequency_penalty": self.frequency_penalty,
                }
            },
            "server": {
                "host": self.host,
                "port": self.port,
                "log_level": self.log_level,
                "timeout": self.timeout,
                "workers": self.workers,
                "batch_size": self.batch_size,
                "max_batch_size": self.max_batch_size,
            },
            "runtime": {
                "tensor_parallel_size": self.tensor_parallel_size,
                "use_trt_llm": self.use_trt_llm,
                "trt_config": {
                    "max_batch_size": self.trt_max_batch_size,
                    "max_beam_width": self.trt_max_beam_width,
                    "max_input_len": self.trt_max_input_len,
                    "max_output_len": self.trt_max_output_len,
                    "build_dir": self.trt_build_dir,
                    "engine_dir": self.trt_engine_dir,
                } if self.use_trt_llm else None,
            },
            "features": {
                "enable_attention_visualization": self.enable_attention_visualization,
                "enable_metrics": self.enable_metrics,
            }
        }

# Example usage
if __name__ == "__main__":
    # Create a default configuration
    config = NIMConfig()
    
    # Print the configuration
    import json
    print(json.dumps(config.to_dict(), indent=2))
    
    # Save the configuration
    config.save("nim_config.json")
    
    # Load the configuration
    loaded_config = NIMConfig.load("nim_config.json")
    
    # Get Docker environment variables
    docker_env_vars = config.get_docker_env_vars()
    
    # Get NIM manifest
    nim_manifest = config.get_nim_manifest()
    
    # Get NIM configuration dictionary
    nim_config_dict = config.get_nim_config_dict()
EOF