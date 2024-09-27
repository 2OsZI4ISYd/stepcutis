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
conda create -n stepcutis python=3.10 -y
source activate stepcutis
pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation


# We now need to install the model repo from huggingface, which is located at
# https://huggingface.co/stepfun-ai/GOT-OCR2_0
# under the tab "files and versions". If this procedure works, then you should see the model
# folder filled with files in the working directory. Otherwise, you will need to find another way.
# If it's not there, then it may either be the case that git-lfs did not install (the two commands below did not run successfully),
# the model repo has been changed/deleted, etc.


curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git-lfs install
git clone https://huggingface.co/stepfun-ai/GOT-OCR2_0

conda deactivate

echo "Setup completed successfully."
