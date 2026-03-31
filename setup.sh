#!/bin/bash
set -e

echo "=== TRIBE v2 Test Setup ==="

# Clone the repo if not already present
if [ ! -d "tribev2" ]; then
    echo "Cloning tribev2..."
    git clone https://github.com/facebookresearch/tribev2.git
else
    echo "tribev2/ already exists, skipping clone"
fi

# Create venv with Python 3.11
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing tribev2..."
pip install -e ./tribev2

echo "Logging into Hugging Face (required for Llama-3.2-3B gated access)..."
huggingface-cli login

echo ""
echo "=== Setup complete ==="
echo "Activate the env:  source .venv/bin/activate"
echo ""
echo "Quick test with text (no video needed):"
echo "  python run_inference.py --text sample_prompt.txt"
echo ""
echo "Test with video:"
echo "  python run_inference.py --video sample.mp4"
