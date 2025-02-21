#!/bin/bash

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Dowload data with Git LFS.
if command_exists git-lfs; then
    echo "Git LFS is installed. Running fetch and pull..."
    git lfs pull
else
    echo "Git LFS is NOT installed. Please install it with 'git lfs install'."
fi

# Add results directory.
DIR="real_data_experiments/results"
if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
    echo "Directory '$DIR' created."
else
    echo "Directory '$DIR' already exists. Nothing to do."
fi
