#!/bin/bash
# SalienceFormer Training Script for RunPod
#
# Usage:
#   ./run_training.sh                    # Default training
#   ./run_training.sh --epochs 5         # Custom epochs

set -e
cd /workspace/SalienceFormer

# Default configuration
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-1e-4}"
DATASET="${DATASET:-wikitext}"
DATASET_CONFIG="${DATASET_CONFIG:-wikitext-2-raw-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/$(date +%Y%m%d_%H%M%S)}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --grad-accum) GRAD_ACCUM="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --epochs N          Number of epochs (default: 3)"
            echo "  --batch-size N      Batch size (default: 8)"
            echo "  --grad-accum N      Gradient accumulation (default: 4)"
            echo "  --lr RATE           Learning rate (default: 1e-4)"
            echo "  --output-dir DIR    Output directory"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "SalienceFormer Training on RunPod"
echo "============================================"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  Output: $OUTPUT_DIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "============================================"

python -c "
import torch
from salienceformer.config import SalienceFormerConfig
from salienceformer.model import SalienceFormer
from salienceformer.train import TrainingArgs, SalienceFormerTrainer, create_dataloaders
from transformers import AutoTokenizer

config = SalienceFormerConfig(
    base_model_name='google/gemma-2b',
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
    device='cuda',
)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
tokenizer.pad_token = tokenizer.eos_token

print('Creating model...')
model = SalienceFormer(config)
print(f'Trainable parameters: {model.get_num_trainable_params():,}')

print('Creating dataloaders...')
train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)

print('Starting training...')
trainer = SalienceFormerTrainer(model, args, tokenizer)
history = trainer.train(train_dataloader, eval_dataloader)

torch.save(model.state_dict(), '$OUTPUT_DIR/final_model.pt')
print(f'Model saved to $OUTPUT_DIR/final_model.pt')
"

echo "Training complete! Outputs: $OUTPUT_DIR"
