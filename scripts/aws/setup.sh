#!/bin/bash
# SalienceFormer AWS EC2 Setup Script
# Run this after launching a Deep Learning AMI instance
#
# Usage: bash setup.sh [--repo-url <git-url>] [--branch <branch>]
#
# Prerequisites:
#   - EC2 instance with Deep Learning AMI (Ubuntu)
#   - GPU instance (g5.xlarge, g6.xlarge, p3.2xlarge recommended)

set -e

# Configuration
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/BrainLLM.git}"
BRANCH="${BRANCH:-main}"
PROJECT_DIR="$HOME/BrainLLM"
VENV_DIR="$HOME/salienceformer-env"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-url) REPO_URL="$2"; shift 2 ;;
        --branch) BRANCH="$2"; shift 2 ;;
        --project-dir) PROJECT_DIR="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "SalienceFormer AWS Setup"
echo "============================================"
echo "Repository: $REPO_URL"
echo "Branch: $BRANCH"
echo "Project directory: $PROJECT_DIR"
echo ""

# System info
echo ">>> System Information"
echo "Instance type: $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# Update system packages
echo ">>> Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq git tmux htop nvtop

# Clone repository
echo ">>> Cloning repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "Directory exists, pulling latest changes..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Create virtual environment
echo ">>> Setting up Python environment..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA (if not already installed via AMI)
echo ">>> Installing/verifying PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SalienceFormer with all dependencies
echo ">>> Installing SalienceFormer..."
cd "$PROJECT_DIR"
pip install -e ".[all]"

# Verify installation
echo ""
echo ">>> Verifying installation..."
python -c "
import torch
import transformers
from salienceformer import SalienceFormer, SalienceFormerConfig

print('Installation verified!')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'  Transformers: {transformers.__version__}')
print(f'  SalienceFormer: ready')
"

# Run tests
echo ""
echo ">>> Running tests..."
pytest tests/ -v --tb=short || echo "Some tests failed (may be expected without full setup)"

# Create convenience scripts
echo ""
echo ">>> Creating convenience scripts..."

cat > "$HOME/activate.sh" << 'EOF'
#!/bin/bash
source ~/salienceformer-env/bin/activate
cd ~/BrainLLM
echo "SalienceFormer environment activated"
echo "Project directory: $(pwd)"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
EOF
chmod +x "$HOME/activate.sh"

# Create data directory
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/outputs"
mkdir -p "$PROJECT_DIR/checkpoints"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo "  source ~/activate.sh"
echo "  python -m salienceformer.train"
echo ""
echo "Or run in tmux for long jobs:"
echo "  tmux new -s train"
echo "  source ~/activate.sh"
echo "  python -m salienceformer.train"
echo "  # Ctrl+B, D to detach"
echo ""
echo "Directories:"
echo "  Project: $PROJECT_DIR"
echo "  Outputs: $PROJECT_DIR/outputs"
echo "  Checkpoints: $PROJECT_DIR/checkpoints"
echo ""
