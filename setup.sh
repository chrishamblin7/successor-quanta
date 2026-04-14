#!/bin/bash
set -e

if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "Creating virtual environment with Python 3.11..."
uv venv .venv --python 3.11

echo "Installing dependencies..."
uv sync

mkdir -p experiments

echo ""
echo "Setup complete. Activate with:"
echo "  source .venv/bin/activate"
