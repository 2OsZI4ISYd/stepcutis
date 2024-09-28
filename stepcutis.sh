#!/usr/bin/env bash

# Function to uninstall stepcutis
uninstall_stepcutis() {
    echo "Uninstalling stepcutis..."

    # Ensure conda commands can be used in the script
    eval "$(conda shell.bash hook)"

    # Remove the conda environment
    conda remove --name stepcutis --all -y

    # Check for the configuration file
    CONFIG_FILE="$HOME/.stepcutis_config"
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        if [ -d "$REPO_DIR" ]; then
            echo "stepcutis repository found at $REPO_DIR"
            read -p "Do you want to remove this directory? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$REPO_DIR"
                echo "Removed stepcutis repository from $REPO_DIR"
            else
                echo "Repository at $REPO_DIR was not removed"
            fi
        else
            echo "stepcutis repository not found at $REPO_DIR"
        fi
        rm "$CONFIG_FILE"
    else
        echo "Could not find stepcutis configuration file"
    fi

    # Remove the stepcutis script
    SCRIPT_PATH=$(which stepcutis)
    if [ -n "$SCRIPT_PATH" ]; then
        sudo rm "$SCRIPT_PATH"
        echo "Removed stepcutis script from $SCRIPT_PATH"
    else
        echo "stepcutis script not found in PATH"
    fi

    echo "stepcutis has been uninstalled."
    exit 0
}

# Check for uninstall command
if [ "$1" = "uninstall" ]; then
    uninstall_stepcutis
fi

# Rest of the script for normal operation
if [ "$#" -ne 2 ]; then
    echo "Usage: stepcutis INPUT_DIR CHUNK_SIZE"
    echo "       stepcutis uninstall"
    exit 1
fi

# Assign arguments to variables
INPUT_DIR="$1"
CHUNK_SIZE="$2"

# Check if INPUT_DIR exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Check if CHUNK_SIZE is a positive integer
if ! [[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: Chunk size must be a positive integer."
    exit 1
fi

# Ensure conda commands can be used in the script
eval "$(conda shell.bash hook)"

# Activate the stepcutis environment
conda activate stepcutis

# Get the repository location
CONFIG_FILE="$HOME/.stepcutis_config"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    SCRIPT_PATH="$REPO_DIR/start.py"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "Error: Cannot find start.py in $REPO_DIR"
        exit 1
    fi
else
    echo "Error: Cannot find stepcutis configuration file."
    exit 1
fi

# Run the stepcutis application
python "$SCRIPT_PATH" --input_dir "$INPUT_DIR" --chunk_size "$CHUNK_SIZE"

# Deactivate the conda environment
conda deactivate

echo "stepcutis application execution completed."
