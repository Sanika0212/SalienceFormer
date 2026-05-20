#!/bin/bash
# SalienceFormer RunPod Setup Script
# Run this after connecting to your RunPod instance
#
# Usage: bash setup.sh

set -e

REPO_URL="${REPO_URL:-https://github.com/Gustav-Proxi/SalienceFormer.git}"
PROJECT_DIR="/workspace/SalienceFormer"

echo "============================================"
echo "SalienceFormer RunPod Setup"
echo "============================================"

# System info
echo ">>> GPU Info"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Clone repository
echo ">>> Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Install dependencies
echo ">>> Installing dependencies..."
pip install -e ".[all]" -q

# Verify installation
echo ""
echo ">>> Verifying installation..."
python -c "
import torch
from salienceformer import SalienceFormer, SalienceFormerConfig

print('Installation verified!')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  cd /workspace/SalienceFormer"
echo "  python -m salienceformer.train"
echo ""
echo "Or run the training script:"
echo "  bash scripts/runpod/run_training.sh"
echo ""
