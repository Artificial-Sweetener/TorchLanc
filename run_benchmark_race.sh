#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the virtual environment directory
VENV_DIR="venv"

# Check if the virtual environment activation script exists.
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "No virtual environment found. Creating one in '$VENV_DIR'..."
    # Create the virtual environment using python3.
    python3 -m venv "$VENV_DIR"

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "Installing TorchLanc in editable mode..."
    pip install -e .

    echo "Installing benchmark-specific dependencies..."
    pip install numpy torch torchvision Pillow py-cpuinfo

    echo "Setup complete."
else
    echo "Found existing virtual environment."
fi


echo ""
echo "=================================================================="
echo "Activating environment and running the benchmark race..."
echo "=================================================================="
echo ""

# Activate the venv (sources again to ensure it's active for this shell session)
# and run the benchmark.
source "$VENV_DIR/bin/activate"
python benchmark/benchmark.py --race

echo ""
echo "Benchmark complete."
