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

# Convert INPUT_DIR to absolute path
INPUT_DIR=$(realpath "$INPUT_DIR")

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

# Get the repository location
CONFIG_FILE="$HOME/.stepcutis_config"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
    if [ ! -d "$REPO_DIR" ]; then
        echo "Error: stepcutis repository directory not found at $REPO_DIR"
        exit 1
    fi
else
    echo "Error: Cannot find stepcutis configuration file."
    exit 1
fi

# Save the current directory
ORIGINAL_DIR=$(pwd)

# Change to the project directory
cd "$REPO_DIR" || { echo "Error: Unable to change to directory $REPO_DIR"; exit 1; }

# Ensure conda commands can be used in the script
eval "$(conda shell.bash hook)"

# Activate the stepcutis environment
conda activate stepcutis

echo "Running stepcutis with INPUT_DIR=$INPUT_DIR and CHUNK_SIZE=$CHUNK_SIZE"

# Run the stepcutis application
python start.py --input_dir "$INPUT_DIR" --chunk_size "$CHUNK_SIZE"

# Capture the exit status of the Python script
PYTHON_EXIT_STATUS=$?

# Deactivate the conda environment
conda deactivate

# Change back to the original directory
cd "$ORIGINAL_DIR" || { echo "Error: Unable to change back to original directory"; exit 1; }

if [ $PYTHON_EXIT_STATUS -eq 0 ]; then
    echo "stepcutis application execution completed successfully."
else
    echo "stepcutis application execution failed with exit status $PYTHON_EXIT_STATUS."
fi

# Exit with the same status as the Python script
exit $PYTHON_EXIT_STATUS
