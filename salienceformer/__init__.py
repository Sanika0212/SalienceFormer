"""
SalienceFormer: Hippocampal Memory Selection for Transformers

A novel memory-augmented transformer architecture that implements
biologically-inspired memory mechanisms:
- Salience Gate: Dual-pathway learned importance scoring (SPW-R analogue)
- Memory Consolidator: Priority-based replay with exponential decay
- Drift Calibrator: Stable retrieval via affine correction

Paper: "SalienceFormer: Hippocampal Memory Selection for Transformers"
Based on: "Selection of experience for memory by hippocampal sharp wave ripples" (Yang et al., 2024)
"""

from salienceformer.config import SalienceFormerConfig
from salienceformer.model import SalienceFormer
from salienceformer.salience.gate import SalienceGate
from salienceformer.memory.buffer import DifferentiablePriorityBuffer
from salienceformer.drift.calibrator import EmbeddingDriftCalibrator
from salienceformer.losses import SalienceFormerLoss, ConsolidationRScore

__all__ = [
    "SalienceFormerConfig",
    "SalienceFormer",
    "SalienceGate",
    "DifferentiablePriorityBuffer",
    "EmbeddingDriftCalibrator",
    "SalienceFormerLoss",
    "ConsolidationRScore",
]
__version__ = "0.1.0"
