#!/bin/bash
# SalienceFormer Training Script for AWS
#
# Usage:
#   ./run_training.sh                    # Default training
#   ./run_training.sh --epochs 5         # Custom epochs
#   ./run_training.sh --dataset pg19     # Different dataset
#   ./run_training.sh --resume ckpt/     # Resume from checkpoint

set -e

# Activate environment
source ~/salienceformer-env/bin/activate
cd ~/BrainLLM

# Default configuration
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
DATASET="${DATASET:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/$(date +%Y%m%d_%H%M%S)}"
BASE_MODEL="${BASE_MODEL:-google/gemma-2b}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
USE_WANDB="${USE_WANDB:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --dataset-config) DATASET_CONFIG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --wandb) USE_WANDB="true"; shift ;;
        --resume) RESUME_FROM="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --epochs N          Number of epochs (default: 3)"
            echo "  --batch-size N      Batch size per GPU (default: 8)"
            echo "  --grad-accum N      Gradient accumulation steps (default: 4)"
            echo "  --lr RATE           Learning rate (default: 1e-4)"
            echo "  --dataset NAME      Dataset name (default: wikitext)"
            echo "  --dataset-config C  Dataset config (default: wikitext-2-raw-v1)"
            echo "  --output-dir DIR    Output directory"
            echo "  --base-model MODEL  Base model (default: google/gemma-2b)"
            echo "  --max-seq-len N     Max sequence length (default: 512)"
            echo "  --wandb             Enable W&B logging"
            echo "  --resume CKPT       Resume from checkpoint"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log configuration
echo "============================================"
echo "SalienceFormer Training"
echo "============================================"
echo "Configuration:"
echo "  Base model: $BASE_MODEL"
echo "  Dataset: $DATASET ($DATASET_CONFIG)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Learning rate: $LR"
echo "  Max sequence length: $MAX_SEQ_LEN"
echo "  Output: $OUTPUT_DIR"
if [ -n "$RESUME_FROM" ]; then
    echo "  Resuming from: $RESUME_FROM"
fi
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "============================================"

# Save config to output dir
cat > "$OUTPUT_DIR/config.json" << EOF
{
    "base_model": "$BASE_MODEL",
    "dataset": "$DATASET",
    "dataset_config": "$DATASET_CONFIG",
    "epochs": $EPOCHS,
    "batch_size": $BATCH_SIZE,
    "gradient_accumulation_steps": $GRAD_ACCUM,
    "learning_rate": $LR,
    "max_seq_length": $MAX_SEQ_LEN,
    "timestamp": "$(date -Iseconds)",
    "instance_type": "$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo 'unknown')"
}
EOF

# Run training
python -c "
import os
import torch
from salienceformer.config import SalienceFormerConfig
from salienceformer.model import SalienceFormer
from salienceformer.train import TrainingArgs, SalienceFormerTrainer, create_dataloaders
from transformers import AutoTokenizer

# Configuration
config = SalienceFormerConfig(
    base_model_name='$BASE_MODEL',
    freeze_base=True,
    use_lora=True,
)

args = TrainingArgs(
    dataset_name='$DATASET',
    dataset_config='$DATASET_CONFIG',
    batch_size=$BATCH_SIZE,
    gradient_accumulation_steps=$GRAD_ACCUM,
    num_epochs=$EPOCHS,
    learning_rate=$LR,
    max_seq_length=$MAX_SEQ_LEN,
    output_dir='$OUTPUT_DIR',
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print('Creating model...')
model = SalienceFormer(config)
print(f'Total parameters: {model.get_num_total_params():,}')
print(f'Trainable parameters: {model.get_num_trainable_params():,}')

print('Creating dataloaders...')
train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)

print('Starting training...')
trainer = SalienceFormerTrainer(model, args, tokenizer)
history = trainer.train(train_dataloader, eval_dataloader)

print('Training complete!')
print(f'Final train loss: {history[\"train_loss\"][-1]:.4f}')
if history['eval_loss']:
    print(f'Final eval loss: {history[\"eval_loss\"][-1]:.4f}')

# Save final model
torch.save(model.state_dict(), '$OUTPUT_DIR/final_model.pt')
print(f'Model saved to $OUTPUT_DIR/final_model.pt')
"

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Outputs saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
