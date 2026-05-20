"""
Differentiable Priority Buffer Module

A fully differentiable priority-based memory buffer that implements multi-round
consolidation replay with exponential decay. This is the learned equivalent of
the ReplayEngine in brain/replay.py.

Key innovations vs. prior work:
- Memorizing Transformers uses non-differentiable kNN retrieval
- MemoryLLM has compress-on-evict but no multi-round replay
- Titans updates via gradient but lacks explicit consolidation rounds
- We implement the biological replay loop with decaying priorities
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiablePriorityBuffer(nn.Module):
    """
    A differentiable priority buffer with multi-round consolidation replay.

    Maintains (key, value, priority, age) tuples and implements:
    1. Priority-based write: Only store if importance > threshold
    2. Eviction: Remove lowest effective priority when buffer full
    3. Multi-round replay: Consolidate high-priority memories with decay
    4. Soft top-k selection: Differentiable approximation for gradient flow

    This mirrors the ReplayEngine.consolidate() logic but is fully differentiable.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        hidden_dim: int = 768,
        decay_rate: float = 0.9,
        max_replay_rounds: int = 10,
        priority_threshold: float = 1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            buffer_size: Maximum number of entries in buffer
            hidden_dim: Dimension of key/value vectors
            decay_rate: Multiplicative decay per replay round (from brain/config.py)
            max_replay_rounds: Maximum consolidation iterations
            priority_threshold: Only store/replay if effective_priority > this
            temperature: Temperature for soft top-k selection
        """
        super().__init__()
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.decay_rate = decay_rate
        self.max_replay_rounds = max_replay_rounds
        self.priority_threshold = priority_threshold
        self.temperature = temperature

        # Buffer storage (registered as buffers for state_dict but not parameters)
        self.register_buffer('keys', torch.zeros(buffer_size, hidden_dim))
        self.register_buffer('values', torch.zeros(buffer_size, hidden_dim))
        self.register_buffer('priorities', torch.ones(buffer_size) * priority_threshold)
        self.register_buffer('ages', torch.zeros(buffer_size))
        self.register_buffer('valid_mask', torch.zeros(buffer_size, dtype=torch.bool))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('total_writes', torch.tensor(0, dtype=torch.long))

        # Learned projections for key/value (optional transformation)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # Query projection for retrieval
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

        # Consolidation output projection
        self.consolidation_proj = nn.Linear(hidden_dim, hidden_dim)

    def compute_effective_priority(self) -> torch.Tensor:
        """
        Compute effective priority: P_eff = P * decay^age

        This is the core equation from ReplayEngine: after each round,
        effective weight drops by decay_rate.

        Returns:
            (buffer_size,) tensor of effective priorities
        """
        return self.priorities * (self.decay_rate ** self.ages)

    def soft_top_k_mask(
        self,
        scores: torch.Tensor,
        k: int,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Differentiable approximation to top-k selection.

        Uses repeated softmax refinement to create a soft mask where
        high-scoring elements approach 1 and low-scoring approach 0.

        Args:
            scores: (N,) scores to select from
            k: Number of elements to select
            temperature: Softmax temperature (lower = sharper)

        Returns:
            (N,) soft mask with approximately k elements near 1
        """
        if temperature is None:
            temperature = self.temperature

        # Normalize scores
        normalized = scores / (temperature + 1e-8)

        # Iterative refinement for sharper selection
        for _ in range(3):
            weights = F.softmax(normalized, dim=-1)
            normalized = normalized + torch.log(weights + 1e-8)

        # Final softmax with low temperature for sharp selection
        weights = F.softmax(normalized / 0.1, dim=-1)

        # Scale so sum approximately equals k
        weights = weights * k

        return weights.clamp(0, 1)

    def write(
        self,
        hidden_states: torch.Tensor,
        importance_weights: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Write salient tokens to buffer based on importance weights.

        Only writes entries where importance_weight > priority_threshold.
        If buffer is full, evicts entry with lowest effective priority.

        Args:
            hidden_states: (batch, seq_len, hidden_dim) tokens to potentially store
            importance_weights: (batch, seq_len) from SalienceGate
            attention_mask: (batch, seq_len) mask for valid tokens

        Returns:
            write_mask: (batch, seq_len) indicating which tokens were written
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        write_mask = torch.zeros(B, T, device=device)

        # Flatten for iteration
        flat_hidden = hidden_states.view(-1, D)
        flat_weights = importance_weights.view(-1)

        if attention_mask is not None:
            flat_mask = attention_mask.view(-1).bool()
        else:
            flat_mask = torch.ones(B * T, dtype=torch.bool, device=device)

        # Find entries to write (importance > threshold and valid)
        write_candidates = (flat_weights > self.priority_threshold) & flat_mask
        candidate_indices = write_candidates.nonzero(as_tuple=True)[0]

        for idx in candidate_indices:
            h = flat_hidden[idx]
            w = flat_weights[idx]

            # Check if buffer has space
            n_valid = self.valid_mask.sum()

            if n_valid < self.buffer_size:
                # Buffer not full: write to next slot
                write_idx = self.write_ptr.item()
                self._write_entry(write_idx, h, w)
                self.write_ptr = (self.write_ptr + 1) % self.buffer_size
            else:
                # Buffer full: evict lowest effective priority if new is higher
                eff_priorities = self.compute_effective_priority()
                eff_priorities = torch.where(
                    self.valid_mask,
                    eff_priorities,
                    torch.tensor(float('inf'), device=device)
                )
                min_idx = eff_priorities.argmin()

                if w > eff_priorities[min_idx]:
                    self._write_entry(min_idx.item(), h, w)

            # Update write mask (reshape back to batch form)
            b_idx = idx // T
            t_idx = idx % T
            write_mask[b_idx, t_idx] = 1.0

            self.total_writes += 1

        return write_mask

    def _write_entry(
        self,
        idx: int,
        hidden: torch.Tensor,
        priority: torch.Tensor,
    ) -> None:
        """Write a single entry to the buffer at position idx."""
        self.keys[idx] = self.key_proj(hidden).detach()
        self.values[idx] = self.value_proj(hidden).detach()
        self.priorities[idx] = priority.detach()
        self.ages[idx] = 0
        self.valid_mask[idx] = True

    def replay_consolidation(
        self,
        query_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform multi-round consolidation replay.

        Each round:
        1. Compute effective priorities (P * decay^age)
        2. Select active memories (eff_priority > threshold)
        3. Retrieve via attention weighted by priority
        4. Apply decay to retrieved memories

        This mirrors ReplayEngine.consolidate() but is differentiable.

        Args:
            query_states: (batch, seq_len, hidden_dim) current context
            attention_mask: (batch, seq_len) mask for valid positions

        Returns:
            consolidated: (batch, hidden_dim) aggregated memory
            replay_log: List of attention weights per round
        """
        B, T, D = query_states.shape
        device = query_states.device

        # Check if buffer has valid entries
        if not self.valid_mask.any():
            return torch.zeros(B, D, device=device), []

        # Project queries
        # Use mean pooling across sequence for global query
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # (B, T, 1)
            query = (query_states * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            query = query_states.mean(dim=1)  # (B, D)

        query = self.query_proj(query)  # (B, D)

        # Working copies of priorities and ages
        working_priorities = self.priorities.clone()
        working_ages = self.ages.clone()

        all_round_outputs = []
        replay_log = []

        for round_idx in range(self.max_replay_rounds):
            # Compute effective priorities
            eff_priorities = working_priorities * (self.decay_rate ** working_ages)

            # Soft mask for "active" memories (eff_priority > threshold)
            # Use sigmoid for differentiability
            active_logits = (eff_priorities - self.priority_threshold) * 10
            active_mask = torch.sigmoid(active_logits) * self.valid_mask.float()

            # Check if any active memories remain
            if active_mask.sum() < 0.5:
                break

            # Attention-based retrieval weighted by effective priority
            # Q: (B, D), K: (M, D) -> attn: (B, M)
            attn_scores = torch.matmul(query, self.keys.T) / math.sqrt(D)

            # Weight by effective priority and active mask
            attn_scores = attn_scores * active_mask.unsqueeze(0) * eff_priorities.unsqueeze(0)

            # Mask out invalid entries
            attn_scores = attn_scores.masked_fill(
                ~self.valid_mask.unsqueeze(0),
                float('-inf')
            )

            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, M)

            # Retrieve values
            round_output = torch.matmul(attn_weights, self.values)  # (B, D)
            all_round_outputs.append(round_output)
            replay_log.append(attn_weights.detach())

            # Update working state: decay priorities of retrieved memories
            # Higher attention weight = more retrieval = more decay
            retrieval_strength = attn_weights.mean(dim=0)  # (M,)
            working_ages = working_ages + retrieval_strength.detach()

        # Age all entries (time passes)
        self.ages = self.ages + 1

        # Combine all rounds with decay weighting (earlier rounds more important)
        if all_round_outputs:
            n_rounds = len(all_round_outputs)
            round_weights = torch.tensor(
                [self.decay_rate ** i for i in range(n_rounds)],
                device=device
            )
            round_weights = round_weights / round_weights.sum()

            consolidated = sum(
                w * out for w, out in zip(round_weights, all_round_outputs)
            )
            consolidated = self.consolidation_proj(consolidated)
        else:
            consolidated = torch.zeros(B, D, device=device)

        return consolidated, replay_log

    def get_buffer_stats(self) -> dict:
        """Get statistics about buffer state."""
        n_valid = self.valid_mask.sum().item()
        eff_priorities = self.compute_effective_priority()
        valid_priorities = eff_priorities[self.valid_mask]

        if n_valid > 0:
            return {
                "n_entries": n_valid,
                "buffer_utilization": n_valid / self.buffer_size,
                "mean_priority": valid_priorities.mean().item(),
                "max_priority": valid_priorities.max().item(),
                "min_priority": valid_priorities.min().item(),
                "mean_age": self.ages[self.valid_mask].mean().item(),
                "total_writes": self.total_writes.item(),
            }
        else:
            return {
                "n_entries": 0,
                "buffer_utilization": 0.0,
                "mean_priority": 0.0,
                "max_priority": 0.0,
                "min_priority": 0.0,
                "mean_age": 0.0,
                "total_writes": self.total_writes.item(),
            }

    def reset(self) -> None:
        """Clear all buffer entries."""
        self.keys.zero_()
        self.values.zero_()
        self.priorities.fill_(self.priority_threshold)
        self.ages.zero_()
        self.valid_mask.zero_()
        self.write_ptr.zero_()
