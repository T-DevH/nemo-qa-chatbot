#!/usr/bin/env python3
"""Debug script for transformer_engine installation and dependencies."""
import os
import sys
import glob
import subprocess
from pathlib import Path

def run_command(cmd):
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def check_cuda_installation():
    """Check CUDA installation and version."""
    print("\n=== CUDA Installation Check ===")
    nvcc_version = run_command("nvcc --version")
    print(f"NVCC Version:\n{nvcc_version}")
    
    cuda_path = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "Not set"))
    print(f"\nCUDA_HOME/CUDA_PATH: {cuda_path}")
    
    if cuda_path != "Not set":
        cuda_version_file = Path(cuda_path) / "version.txt"
        if cuda_version_file.exists():
            print(f"CUDA version from version.txt: {cuda_version_file.read_text().strip()}")
        
        # Check for CUDA libraries
        cuda_lib_path = Path(cuda_path) / "lib64"
        if cuda_lib_path.exists():
            cuda_libs = list(cuda_lib_path.glob("libcudart.so*"))
            print(f"CUDA libraries found: {[lib.name for lib in cuda_libs]}")

def check_transformer_engine():
    """Check transformer_engine installation and dependencies."""
    print("\n=== Transformer Engine Check ===")
    
    try:
        import transformer_engine
        print(f"Transformer-engine version: {transformer_engine.__version__}")
        print(f"Transformer-engine package path: {os.path.dirname(transformer_engine.__file__)}")
        
        # Check for binary files
        pytorch_dir = os.path.join(os.path.dirname(transformer_engine.__file__), "pytorch")
        print(f"\nLooking for binary files in: {pytorch_dir}")
        
        if os.path.exists(pytorch_dir):
            files = os.listdir(pytorch_dir)
            print(f"Files in pytorch dir: {files}")
            
            # Look for .so files
            so_files = glob.glob(os.path.join(pytorch_dir, "*.so*"))
            print(f"SO files: {so_files}")
            
            if not so_files:
                print("WARNING: No .so files found! This indicates a potential installation issue.")
        else:
            print(f"ERROR: Directory {pytorch_dir} does not exist!")
        
        # Try to import TransformerLayer
        print("\nAttempting to import TransformerLayer...")
        try:
            from transformer_engine.pytorch import TransformerLayer
            print("Successfully imported TransformerLayer")
        except Exception as e:
            print(f"Error importing TransformerLayer: {e}")
            import traceback
            traceback.print_exc()
            
    except ImportError as e:
        print(f"Error importing transformer_engine: {e}")
        import traceback
        traceback.print_exc()

def check_pytorch_cuda():
    """Check PyTorch CUDA configuration."""
    print("\n=== PyTorch CUDA Check ===")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
            
            # Test CUDA tensor operations
            print("\nTesting CUDA tensor operations...")
            x = torch.rand(2, 3).cuda()
            y = torch.rand(2, 3).cuda()
            z = x + y
            print("CUDA tensor operation successful!")
            
    except Exception as e:
        print(f"Error checking PyTorch CUDA: {e}")
        import traceback
        traceback.print_exc()

def check_environment():
    """Check Python environment and paths."""
    print("\n=== Environment Check ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    
    # Check pip packages
    print("\nInstalled packages:")
    pip_list = run_command(f"{sys.executable} -m pip list")
    print(pip_list)

def main():
    """Run all checks."""
    print("Starting transformer_engine debug...")
    
    check_environment()
    check_cuda_installation()
    check_pytorch_cuda()
    check_transformer_engine()
    
    print("\nDebug complete!")

if __name__ == "__main__":
    main()