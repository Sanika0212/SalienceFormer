<h1 align="center">SalienceFormer</h1>

<p align="center">
  <em>Hippocampal memory selection for transformers — biologically-inspired consolidation for large language models</em>
</p>

<p align="center">
  <a href="https://huggingface.co/Gustav-Proxi/SalienceFormer-Gemma2B"><img src="https://img.shields.io/badge/HuggingFace-Model-FFD21E?style=flat-square&logo=huggingface" alt="HuggingFace"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/WikiText--2_PPL-11.83-58A6FF?style=flat-square" alt="PPL 11.83">
</p>

<p align="center">
  <a href="#architecture">Architecture</a> &bull;
  <a href="#results">Results</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#usage">Usage</a> &bull;
  <a href="#citation">Citation</a>
</p>

---

## Overview

**SalienceFormer** integrates hippocampal memory mechanisms directly into transformer architectures. Inspired by how the human hippocampus selectively consolidates important memories through Sharp Wave Ripples (SPW-Rs), SalienceFormer learns to:

- **Selectively tag** important tokens — like the brain identifies significant events via SPW-Rs
- **Consolidate memories** through priority-based replay — like sleep-phase consolidation
- **Maintain stable representations** through drift calibration — analogous to synaptic homeostasis

Three lightweight learned modules wrap a frozen Gemma-2B base and add only ~15M trainable parameters while achieving **11.83 PPL on WikiText-2**.

---

## Architecture

<p align="center">
  <img src="docs/figures/architecture.png" alt="SalienceFormer Architecture" width="720">
</p>

The forward pass is a six-stage pipeline through four specialized modules:

| Stage | Module | Brain Analogue |
|-------|--------|----------------|
| 1. Hidden states | Gemma-2B (frozen + LoRA) | Cortex |
| 2. Importance scoring | Salience Gate (dual-path MLP + cross-attn) | Sharp Wave Ripples |
| 3. Drift correction | Drift Calibrator (learned affine: h′ = Ah + b) | Synaptic homeostasis |
| 4. Priority storage | Memory Consolidator (priority buffer + multi-round replay) | Sleep replay |
| 5. Fusion | Output Fusion (cross-attention + gating) | Memory retrieval |
| 6. Prediction | LM Head | — |

### Salience Gate

Dual-pathway importance scoring — local token-intrinsic MLP combined with a global cross-attention pathway, producing importance weights in [1.0, 5.0]:

```python
local_scores  = MLP(hidden_states)          # token-intrinsic (single-electrode ripple)
global_scores = CrossAttention(hidden_states) # population synchrony
salience      = sigmoid(w * local + (1-w) * global - threshold)
```

### Memory Consolidator

Priority-based buffer with multi-round replay consolidation and exponential decay (γ = 0.9):

```python
buffer.store(keys, values, priorities)       # priority = salience × importance_weight
for round in range(max_rounds):
    consolidated = replay(buffer, decay_rate ** round)
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | 2048 | Memory buffer capacity |
| `decay_rate` | 0.9 | Consolidation decay per round |
| `importance_range` | [1.0, 5.0] | Min/max importance weights |
| `salience_threshold` | 0.0 | Initial threshold (learned) |

---

## Results

### Perplexity on WikiText-2

| Model | Parameters | WikiText-2 PPL |
|-------|------------|----------------|
| GPT-2 | 124M | 29.41 |
| Gemma-2B | 2B | ~18 |
| **SalienceFormer** | 2B + 15M | **11.83** |

### Ablation Study

<p align="center">
  <img src="docs/figures/ablation_study.png" alt="Ablation Study: Component Contributions" width="720">
</p>

Both hippocampal components are essential — removing either collapses performance:

| Configuration | PPL | Δ PPL |
|--------------|-----|-------|
| Full SalienceFormer | **11.83** | — |
| Without Salience Gate | 39.75 | +27.92 |
| Without Memory Buffer | 89.84 | +78.01 |
| Random Salience | 89.84 | +78.01 |

### Brain-Like Behavior

<p align="center">
  <img src="docs/figures/salience_heatmap.png" alt="Salience Gate: Selective Token Tagging" width="720">
</p>

The salience gate exhibits selectivity consistent with hippocampal tagging — content words receive systematically higher salience than function words:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Content/Function Word Ratio | **2.11x** | Content words tagged more (selective memory) |
| Long-Range PPL Benefit | **+6.95** | Better on late tokens (remembers context) |
| Buffer Priority | **4.9/5.0** | High-importance items retained |
| Temporal Coherence | **0.58** | Nearby tokens tagged together |

<p align="center">
  <img src="docs/figures/results_summary.png" alt="Key Results Summary" width="680">
</p>

---

## Installation

```bash
git clone https://github.com/Gustav-Proxi/SalienceFormer.git
cd SalienceFormer

# Install with training dependencies
pip install -e ".[train]"

# Or install everything (dev + train + eval)
pip install -e ".[all]"
```

**Requirements:** Python 3.10+, PyTorch 2.0+, Transformers 4.36+, CUDA 11.8+ (for GPU)

---

## Usage

### Quick Start

```python
from salienceformer import SalienceFormer, SalienceFormerConfig
from transformers import AutoTokenizer
import torch

config = SalienceFormerConfig(
    base_model_name="google/gemma-2b",
    freeze_base=True,
    use_lora=True,
)
model = SalienceFormer(config)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

# Load pretrained weights
ckpt = torch.load("pytorch_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)

# Generate
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

### Load from HuggingFace

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="Gustav-Proxi/SalienceFormer-Gemma2B",
    filename="pytorch_model.pt"
)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"], strict=False)
```

### Training

```bash
python -m salienceformer.train \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --batch_size 8 \
    --num_epochs 3 \
    --output_dir ./outputs
```

### Evaluation

```bash
python -m evaluation.comprehensive_eval \
    --checkpoint ./outputs/checkpoint-step-110000/checkpoint.pt \
    --output results.json \
    --device cuda
```

---

## Neuroscience Background

| Brain Mechanism | SalienceFormer Implementation |
|-----------------|-------------------------------|
| Sharp Wave Ripples (SPW-Rs) | Salience Gate — dual-pathway importance scoring |
| Memory tagging | Importance weights [1.0, 5.0] |
| Sleep replay | Multi-round consolidation with exponential decay |
| Synaptic homeostasis | Drift Calibrator — learned affine correction |

**Key insight:** The hippocampus does not remember everything equally. It selectively tags important experiences and replays them during sleep. SalienceFormer brings this mechanism to transformer hidden states.

---

## Project Structure

```
SalienceFormer/
├── salienceformer/
│   ├── config.py              # SalienceFormerConfig
│   ├── model.py               # SalienceFormer (main model)
│   ├── train.py               # Training script
│   ├── losses.py              # Multi-objective losses
│   ├── salience/gate.py       # SalienceGate
│   ├── memory/buffer.py       # DifferentiablePriorityBuffer
│   └── drift/calibrator.py    # EmbeddingDriftCalibrator
├── evaluation/
│   ├── comprehensive_eval.py  # Full evaluation suite
│   ├── ablation.py            # Ablation framework
│   ├── metrics.py             # PPL, BLEU, ROUGE, F1
│   └── visualization.py       # Paper figures
├── docs/figures/              # Architecture, results, and heatmap figures
├── scripts/
│   ├── runpod/                # Cloud training scripts
│   └── aws/                   # AWS deployment
└── tests/                     # Unit tests
```

---

## Citation

```bibtex
@misc{salienceformer2025,
  title={SalienceFormer: Hippocampal Memory Selection for Transformers},
  author={Vaishak Girish Kumar and Sanika},
  year={2025},
  howpublished={\url{https://github.com/Gustav-Proxi/SalienceFormer}},
}
```

---

## Contributors

- **Vaishak Girish Kumar** — [github.com/Gustav-Proxi](https://github.com/Gustav-Proxi)
- **Sanika** — [github.com/Sanika0212](https://github.com/Sanika0212)

---

## Acknowledgments

- Built on [Gemma](https://ai.google.dev/gemma) by Google DeepMind
- Inspired by hippocampal Sharp Wave Ripple research
- Training infrastructure on [RunPod](https://runpod.io)

---

<p align="center">
  <strong>SalienceFormer</strong> — Bringing biological memory to artificial intelligence
</p>
