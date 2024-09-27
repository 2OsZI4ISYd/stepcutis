#!/bin/bash

# Function to check if a command exists
check_command() {
    if command -v $1 &> /dev/null; then
        echo "$1 is installed."
    else
        echo "$1 could not be found. Please install it, and run this setup again."
        exit 1
    fi
}

# Check for required tools
check_command conda
check_command python
check_command nvcc

echo "All required tools are installed."

# Proceed with environment setup and package installation
conda create -n LakeOCR python=3.10 -y
conda activate LakeOCR
pip install -r requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation
git lfs install
git clone https://huggingface.co/stepfun-ai/GOT-OCR2_0
conda deactivate

echo "Setup completed successfully."