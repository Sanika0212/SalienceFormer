"""
SalienceFormer: Hippocampal Memory Selection for Transformers

Main model class that wraps a pretrained transformer with biologically-inspired
memory mechanisms:
1. Salience Gate - learned importance scoring (SPW-R analogue)
2. Memory Consolidator - priority-based replay buffer
3. Drift Calibrator - stable retrieval via affine correction
"""

import math
from typing import Optional, Dict, Any, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoConfig

from salienceformer.config import SalienceFormerConfig
from salienceformer.salience.gate import SalienceGate
from salienceformer.memory.buffer import DifferentiablePriorityBuffer
from salienceformer.drift.calibrator import EmbeddingDriftCalibrator


class OutputFusion(nn.Module):
    """
    Fuses consolidated memory with current hidden states via cross-attention + gating.

    Query: current hidden states
    Key/Value: consolidated memory
    Output: gated combination of memory-attended and original states
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Cross-attention: Q from hidden states, K/V from memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating mechanism: decides how much to use memory vs. original
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse memory with hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            memory: (batch, hidden_dim) consolidated memory
            attention_mask: (batch, seq_len) optional mask

        Returns:
            fused: (batch, seq_len, hidden_dim)
        """
        B, T, D = hidden_states.shape

        # Expand memory to match sequence length for cross-attention
        # Memory becomes key/value: (batch, 1, hidden_dim)
        memory_expanded = memory.unsqueeze(1)

        # Cross-attention: query from hidden_states, key/value from memory
        attn_output, _ = self.cross_attn(
            query=hidden_states,
            key=memory_expanded.expand(-1, T, -1),
            value=memory_expanded.expand(-1, T, -1),
        )

        # Compute gate value
        gate_input = torch.cat([hidden_states, attn_output], dim=-1)
        gate_value = self.gate(gate_input)  # (B, T, 1)

        # Gated fusion
        fused = gate_value * attn_output + (1 - gate_value) * hidden_states

        # Layer norm
        fused = self.norm(fused)

        return fused


class SalienceFormer(nn.Module):
    """
    SalienceFormer: Hippocampal Memory Selection for Transformers

    Wraps a pretrained transformer with biologically-inspired memory mechanisms.
    Can be used with any HuggingFace causal language model as the base.

    Architecture:
    1. Base Transformer (frozen or with LoRA) produces hidden states
    2. Salience Gate scores token importance
    3. Drift Calibrator corrects for distribution shift
    4. Memory Consolidator stores/retrieves priority memories
    5. Output Fusion blends memory with hidden states
    6. LM Head produces output logits
    """

    def __init__(
        self,
        config: SalienceFormerConfig,
        base_model: Optional[PreTrainedModel] = None,
    ):
        """
        Args:
            config: SalienceFormerConfig with all hyperparameters
            base_model: Optional pretrained model. If None, loads from config.base_model_name
        """
        super().__init__()
        self.config = config

        # Load base model if not provided
        if base_model is None:
            # Use float32 by default to avoid dtype mismatches with SalienceFormer modules
            # Users can pass a float16 base_model explicitly if needed for memory savings
            self.base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        else:
            self.base_model = base_model

        # Get hidden dimension from base model
        base_config = self.base_model.config
        self.hidden_dim = getattr(base_config, 'hidden_size', config.hidden_dim)
        self.vocab_size = base_config.vocab_size

        # Optionally freeze base model
        if config.freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Optionally add LoRA adapters
        if config.use_lora:
            self._add_lora_adapters()

        # === Core SalienceFormer Modules ===

        # Salience Gate (SPW-R detector)
        self.salience_gate = SalienceGate(
            hidden_dim=self.hidden_dim,
            n_heads=config.salience_n_heads,
            min_duration=config.salience_min_duration,
            threshold_init=config.salience_threshold_init,
            importance_range=config.importance_weight_range,
            untagged_weight=config.untagged_weight,
        )

        # Memory Consolidator (Replay Engine)
        self.memory_consolidator = DifferentiablePriorityBuffer(
            buffer_size=config.buffer_size,
            hidden_dim=self.hidden_dim,
            decay_rate=config.decay_rate,
            max_replay_rounds=config.max_replay_rounds,
            priority_threshold=config.priority_threshold,
            temperature=config.soft_topk_temperature,
        )

        # Drift Calibrator
        self.drift_calibrator = EmbeddingDriftCalibrator(
            hidden_dim=self.hidden_dim,
            n_anchors=config.n_anchors,
            drift_threshold=config.drift_threshold,
            update_momentum=config.drift_update_momentum,
        )

        # Output Fusion
        self.output_fusion = OutputFusion(
            hidden_dim=self.hidden_dim,
            n_heads=config.fusion_n_heads,
            dropout=config.fusion_dropout,
        )

        # LM Head (try to reuse from base model)
        if hasattr(self.base_model, 'lm_head'):
            self.lm_head = self.base_model.lm_head
        else:
            self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def _add_lora_adapters(self) -> None:
        """Add LoRA adapters to base model attention layers."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=list(self.config.lora_target_modules),
                task_type=TaskType.CAUSAL_LM,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
        except ImportError:
            print("Warning: peft not installed. Skipping LoRA adapters.")
            print("Install with: pip install peft")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        return_salience: bool = False,
        return_memory_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hippocampal memory mechanisms.

        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: (batch, seq_len) attention mask
            labels: (batch, seq_len) labels for language modeling loss
            return_dict: Always True (for compatibility)
            return_salience: Include salience scores in output
            return_memory_stats: Include memory/drift statistics

        Returns:
            Dictionary with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar (if labels provided)
                - salience_scores: (batch, seq_len) (if return_salience)
                - importance_weights: (batch, seq_len) (if return_salience)
                - memory_stats: dict (if return_memory_stats)
                - drift_stats: dict (if return_memory_stats)
        """
        # 1. Get base model hidden states
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last hidden state
        if hasattr(base_outputs, 'hidden_states'):
            hidden_states = base_outputs.hidden_states[-1]
        else:
            # Fallback for models without hidden_states output
            hidden_states = base_outputs.last_hidden_state

        hidden_states = hidden_states.float()  # Ensure float32 for our modules

        # 2. Compute salience scores (SPW-R detection)
        salience_scores, importance_weights = self.salience_gate(
            hidden_states,
            attention_mask=attention_mask,
        )

        # 3. Apply drift correction
        corrected_states, drift_mag = self.drift_calibrator(
            hidden_states,
            update_stats=self.training,
        )

        # 4. Write salient tokens to memory
        if self.training:
            self.memory_consolidator.write(
                corrected_states,
                importance_weights,
                attention_mask=attention_mask,
            )

        # 5. Perform consolidation replay and retrieve
        consolidated_memory, replay_log = self.memory_consolidator.replay_consolidation(
            corrected_states,
            attention_mask=attention_mask,
        )

        # 6. Fuse memory with current hidden states
        if consolidated_memory.abs().sum() > 0:
            fused_states = self.output_fusion(
                corrected_states,
                consolidated_memory,
                attention_mask=attention_mask,
            )
        else:
            fused_states = corrected_states

        # 7. Project to vocabulary
        # Cast to match lm_head dtype (may be float16 from base model)
        lm_head_dtype = next(self.lm_head.parameters()).dtype
        logits = self.lm_head(fused_states.to(lm_head_dtype))

        # Build output dictionary
        outputs = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self._compute_loss(
                logits=logits,
                labels=labels,
                salience_scores=salience_scores,
                importance_weights=importance_weights,
                drift_mag=drift_mag,
            )
            outputs["loss"] = loss

        if return_salience:
            outputs["salience_scores"] = salience_scores
            outputs["importance_weights"] = importance_weights

        if return_memory_stats:
            outputs["memory_stats"] = self.memory_consolidator.get_buffer_stats()
            outputs["drift_stats"] = self.drift_calibrator.get_drift_stats()
            outputs["salience_stats"] = self.salience_gate.get_salience_stats(
                salience_scores, attention_mask
            )

        return outputs

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        salience_scores: torch.Tensor,
        importance_weights: torch.Tensor,
        drift_mag: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-objective loss.

        Components:
        1. Standard LM loss
        2. Salience-weighted LM loss (higher weight for salient tokens)
        3. Sparsity regularization (encourage selective tagging)
        4. Drift regularization (prevent excessive correction)
        """
        # Shift for next-token prediction
        # Cast to float32 for stable loss computation
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        shift_salience = salience_scores[..., :-1].contiguous().float()
        shift_weights = importance_weights[..., :-1].contiguous().float()

        # 1. Standard LM loss
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='mean',
        )

        # 2. Salience-weighted LM loss
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        ).view_as(shift_labels)

        # Weight by importance (normalize to maintain scale)
        weight_norm = shift_weights.sum() / (shift_weights.numel() + 1e-8)
        weighted_lm_loss = (token_losses * shift_weights).mean() / (weight_norm + 1e-8)

        # 3. Sparsity regularization (binary entropy)
        # Encourages salience to be either 0 or 1, not uniform
        eps = 1e-8
        sparsity_loss = -(
            shift_salience * torch.log(shift_salience + eps) +
            (1 - shift_salience) * torch.log(1 - shift_salience + eps)
        ).mean()

        # 4. Drift regularization
        drift_reg = self.drift_calibrator.regularization_loss()

        # Combine losses with configured weights
        total_loss = (
            self.config.lm_loss_weight * lm_loss +
            self.config.salience_weighted_loss_weight * weighted_lm_loss +
            self.config.sparsity_loss_weight * sparsity_loss +
            0.001 * drift_reg  # Small weight for drift regularization
        )

        return total_loss

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text with hippocampal memory augmentation.

        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: (batch, seq_len) attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs. greedy)

        Returns:
            generated_ids: (batch, seq_len + max_new_tokens)
        """
        self.eval()

        batch_size = input_ids.size(0)
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        generated = input_ids.clone()
        gen_mask = attention_mask.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=generated,
                    attention_mask=gen_mask,
                )

                # Get next token logits
                next_logits = outputs["logits"][:, -1, :] / temperature

                # Apply top-p (nucleus) sampling
                if do_sample and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float('-inf')

                # Sample or greedy decode
                if do_sample:
                    probs = F.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=-1)
                gen_mask = torch.cat([gen_mask, torch.ones(batch_size, 1, device=device)], dim=-1)

                # Check for EOS (simplified - would need tokenizer for proper check)
                # For now, just generate max_new_tokens

        return generated

    def reset_memory(self) -> None:
        """Reset memory buffer and drift calibrator."""
        self.memory_consolidator.reset()
        self.drift_calibrator.reset_anchors()

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
