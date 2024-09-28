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

# Ensure conda commands can be used in the script
eval "$(conda shell.bash hook)"

# Check if stepcutis is already installed
if command -v stepcutis &> /dev/null; then
    echo "stepcutis is already installed. Please uninstall it first using 'stepcutis uninstall'."
    exit 1
fi

# Proceed with environment setup and package installation
conda create -n stepcutis python=3.10 -y
conda activate stepcutis

pip install -r requirements.txt
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn==2.6.3 --no-build-isolation

# Install git-lfs and clone the model repo
if ! command -v git-lfs &> /dev/null; then
    echo "Installing git-lfs..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs
    git-lfs install
else
    echo "git-lfs is already installed."
fi

# Clone the model repo
git clone https://huggingface.co/stepfun-ai/GOT-OCR2_0

# Clone the stepcutis repository
REPO_DIR="$HOME/stepcutis"
git clone https://github.com/2OsZI4ISYd/stepcutis.git "$REPO_DIR"

# Store the repository location
echo "$REPO_DIR" > "$HOME/.stepcutis_repo_location"

# Make stepcutis globally accessible
INSTALL_DIR="/usr/local/bin"
echo "Installing stepcutis script to $INSTALL_DIR"
sudo cp "$REPO_DIR/stepcutis" "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR/stepcutis"

# Deactivate the conda environment
conda deactivate

echo "Setup completed successfully."
echo "You can now run 'stepcutis INPUT_DIR CHUNK_SIZE' from any directory."
echo "To uninstall, run 'stepcutis uninstall'."
