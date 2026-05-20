#!/bin/bash
# SalienceFormer Evaluation Script for AWS
#
# Usage:
#   ./run_evaluation.sh                          # Default evaluation
#   ./run_evaluation.sh --checkpoint ckpt/       # Evaluate checkpoint
#   ./run_evaluation.sh --ablation               # Run ablation study
#   ./run_evaluation.sh --datasets wikitext pg19 # Multiple datasets

set -e

# Activate environment
source ~/salienceformer-env/bin/activate
cd ~/BrainLLM

# Default configuration
CHECKPOINT="${CHECKPOINT:-}"
DATASETS="${DATASETS:-wikitext-2}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/eval_$(date +%Y%m%d_%H%M%S)}"
RUN_ABLATION="${RUN_ABLATION:-false}"
ABLATION_SEEDS="${ABLATION_SEEDS:-3}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --datasets)
            DATASETS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                DATASETS="$DATASETS $1"
                shift
            done
            DATASETS="${DATASETS# }"
            ;;
        --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --ablation) RUN_ABLATION="true"; shift ;;
        --ablation-seeds) ABLATION_SEEDS="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [options]"
            echo "  --checkpoint PATH   Model checkpoint to evaluate"
            echo "  --datasets D1 D2    Datasets to evaluate on"
            echo "                      Options: wikitext-2, wikitext-103, pg19, narrativeqa"
            echo "  --max-samples N     Max samples per dataset (default: 1000)"
            echo "  --output-dir DIR    Output directory"
            echo "  --ablation          Run ablation study"
            echo "  --ablation-seeds N  Number of seeds for ablation (default: 3)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "SalienceFormer Evaluation"
echo "============================================"
echo "Configuration:"
echo "  Datasets: $DATASETS"
echo "  Max samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_DIR"
if [ -n "$CHECKPOINT" ]; then
    echo "  Checkpoint: $CHECKPOINT"
fi
if [ "$RUN_ABLATION" = "true" ]; then
    echo "  Ablation: enabled ($ABLATION_SEEDS seeds)"
fi
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""
echo "============================================"

# Run standard evaluation
echo ">>> Running evaluation pipeline..."
python -m evaluation.runner \
    --datasets $DATASETS \
    --max-samples "$MAX_SAMPLES" \
    --output-dir "$OUTPUT_DIR" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"}

# Run ablation study if requested
if [ "$RUN_ABLATION" = "true" ]; then
    echo ""
    echo ">>> Running ablation study..."
    python -c "
import json
from pathlib import Path
from evaluation.ablation import AblationRunner, ABLATION_VARIANTS
from salienceformer.config import SalienceFormerConfig

output_dir = Path('$OUTPUT_DIR') / 'ablation'
output_dir.mkdir(exist_ok=True)

# Base configuration
base_config = SalienceFormerConfig(
    base_model_name='google/gemma-2b',
    freeze_base=True,
    use_lora=True,
)

runner = AblationRunner(
    base_config=base_config,
    output_dir=str(output_dir),
    num_seeds=$ABLATION_SEEDS,
)

# Run subset of ablations (full set takes a long time)
variants_to_run = [
    'baseline',
    'no_salience',
    'no_memory',
    'no_drift',
    'salience_local_only',
    'salience_global_only',
]

print(f'Running ablation variants: {variants_to_run}')
results = runner.run_ablation_suite(variants_to_run)

# Save results
with open(output_dir / 'ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'Ablation results saved to {output_dir}')
runner.generate_comparison_table()
"
fi

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
