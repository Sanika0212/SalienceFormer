"""
Visualization Utilities for SalienceFormer

Plotting functions for salience, memory dynamics, and experiment results.
"""

import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import torch

# Lazy imports for plotting libraries
plt = None
sns = None


def _ensure_matplotlib():
    """Lazily import matplotlib."""
    global plt
    if plt is None:
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt


def _ensure_seaborn():
    """Lazily import seaborn."""
    global sns
    if sns is None:
        try:
            import seaborn as _sns
            sns = _sns
        except ImportError:
            sns = None
    return sns


def setup_plotting_style():
    """Set up consistent plotting style for paper figures."""
    plt = _ensure_matplotlib()
    sns = _ensure_seaborn()

    # Paper-ready style
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    if sns:
        sns.set_palette("colorblind")


def plot_salience_heatmap(
    tokens: List[str],
    salience_scores: Union[np.ndarray, torch.Tensor],
    importance_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Token Salience Scores",
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
    show_colorbar: bool = True,
    highlight_threshold: float = 0.5,
) -> Any:
    """
    Create a heatmap visualization of salience scores over tokens.

    Args:
        tokens: List of token strings
        salience_scores: Salience scores [seq_len] or [batch, seq_len]
        importance_weights: Optional importance weights for annotation
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show_colorbar: Whether to show colorbar
        highlight_threshold: Threshold above which to highlight tokens

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()
    sns = _ensure_seaborn()

    # Convert to numpy
    if isinstance(salience_scores, torch.Tensor):
        salience_scores = salience_scores.detach().cpu().numpy()

    if salience_scores.ndim == 2:
        salience_scores = salience_scores[0]  # Take first batch item

    if importance_weights is not None:
        if isinstance(importance_weights, torch.Tensor):
            importance_weights = importance_weights.detach().cpu().numpy()
        if importance_weights.ndim == 2:
            importance_weights = importance_weights[0]

    # Truncate if necessary
    n_tokens = min(len(tokens), len(salience_scores))
    tokens = tokens[:n_tokens]
    salience_scores = salience_scores[:n_tokens]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap data (reshape for imshow)
    heatmap_data = salience_scores.reshape(1, -1)

    # Plot
    if sns:
        sns.heatmap(
            heatmap_data,
            xticklabels=tokens,
            yticklabels=[""],
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            cbar=show_colorbar,
            ax=ax,
        )
    else:
        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right")
        ax.set_yticks([])
        if show_colorbar:
            plt.colorbar(im, ax=ax)

    # Highlight high-salience tokens
    for i, (token, score) in enumerate(zip(tokens, salience_scores)):
        if score >= highlight_threshold:
            ax.axvline(x=i, color="red", alpha=0.3, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Tokens")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved salience heatmap to {save_path}")

    return fig


def plot_memory_dynamics(
    buffer_history: List[Dict[str, Any]],
    title: str = "Memory Buffer Dynamics",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> Any:
    """
    Plot memory buffer dynamics over time/steps.

    Args:
        buffer_history: List of buffer stats dicts with keys:
            - 'step': training step
            - 'n_entries': number of entries in buffer
            - 'mean_priority': average priority
            - 'max_priority': maximum priority
            - 'buffer_utilization': fraction of buffer used

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    steps = [h.get("step", i) for i, h in enumerate(buffer_history)]

    # Buffer utilization
    ax = axes[0, 0]
    utilization = [h.get("buffer_utilization", 0) for h in buffer_history]
    ax.plot(steps, utilization, "b-", linewidth=2)
    ax.fill_between(steps, utilization, alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Buffer Utilization")
    ax.set_title("Memory Buffer Utilization")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Full capacity")
    ax.legend()

    # Number of entries
    ax = axes[0, 1]
    n_entries = [h.get("n_entries", 0) for h in buffer_history]
    ax.plot(steps, n_entries, "g-", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Number of Entries")
    ax.set_title("Buffer Entries Over Time")

    # Priority statistics
    ax = axes[1, 0]
    mean_priority = [h.get("mean_priority", 0) for h in buffer_history]
    max_priority = [h.get("max_priority", 0) for h in buffer_history]
    ax.plot(steps, mean_priority, "b-", label="Mean Priority", linewidth=2)
    ax.plot(steps, max_priority, "r-", label="Max Priority", linewidth=2)
    ax.fill_between(steps, mean_priority, alpha=0.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Priority")
    ax.set_title("Memory Priorities")
    ax.legend()

    # Priority distribution (last step)
    ax = axes[1, 1]
    if "priority_distribution" in buffer_history[-1]:
        priorities = buffer_history[-1]["priority_distribution"]
        ax.hist(priorities, bins=30, color="purple", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Priority Value")
        ax.set_ylabel("Count")
        ax.set_title("Final Priority Distribution")
    else:
        ax.text(0.5, 0.5, "No distribution data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Priority Distribution (N/A)")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved memory dynamics plot to {save_path}")

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    smoothing: float = 0.0,
) -> Any:
    """
    Plot training and evaluation curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'eval_loss', etc.
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        smoothing: Exponential smoothing factor (0 = none, 0.9 = heavy)

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()

    def smooth(values, factor):
        if factor <= 0:
            return values
        smoothed = []
        last = values[0] if values else 0
        for v in values:
            last = factor * last + (1 - factor) * v
            smoothed.append(last)
        return smoothed

    # Determine subplot layout
    n_plots = len(history)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (name, values) in enumerate(history.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        steps = list(range(len(values)))

        # Plot raw values with transparency
        ax.plot(steps, values, alpha=0.3, color=colors[idx % 10])

        # Plot smoothed values
        smoothed = smooth(values, smoothing)
        ax.plot(steps, smoothed, color=colors[idx % 10], linewidth=2, label=name)

        ax.set_xlabel("Step")
        ax.set_ylabel(name.replace("_", " ").title())
        ax.set_title(name.replace("_", " ").title())
        ax.legend()

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved training curves to {save_path}")

    return fig


def plot_ablation_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "perplexity",
    title: str = "Ablation Study Results",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    sort_by_value: bool = True,
    show_error_bars: bool = True,
) -> Any:
    """
    Create bar chart comparing ablation experiment results.

    Args:
        results: Dictionary mapping variant name to metrics dict
                 Expected format: {'variant': {'metric_mean': x, 'metric_std': y}}
        metric: Which metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        sort_by_value: Sort bars by metric value
        show_error_bars: Show standard deviation error bars

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()
    sns = _ensure_seaborn()

    # Extract data
    names = []
    values = []
    errors = []

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    for name, metrics in results.items():
        if mean_key in metrics:
            names.append(name)
            values.append(metrics[mean_key])
            errors.append(metrics.get(std_key, 0))

    if not names:
        # Try direct metric values
        for name, metrics in results.items():
            if metric in metrics:
                names.append(name)
                values.append(metrics[metric])
                errors.append(0)

    if not names:
        raise ValueError(f"No data found for metric: {metric}")

    # Sort by value
    if sort_by_value:
        sorted_idx = np.argsort(values)
        names = [names[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        errors = [errors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar chart
    x = np.arange(len(names))
    bars = ax.barh(x, values, xerr=errors if show_error_bars else None,
                   capsize=3, color="steelblue", alpha=0.8)

    # Highlight best result
    best_idx = np.argmin(values) if "loss" in metric or "perplexity" in metric else np.argmax(values)
    bars[best_idx].set_color("forestgreen")

    ax.set_yticks(x)
    ax.set_yticklabels(names)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title)

    # Add value labels
    for i, (v, e) in enumerate(zip(values, errors)):
        label = f"{v:.2f}"
        if show_error_bars and e > 0:
            label += f" ± {e:.2f}"
        ax.text(v + max(values) * 0.01, i, label, va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved ablation comparison to {save_path}")

    return fig


def plot_perplexity_by_position(
    position_ppl: Dict[int, float],
    title: str = "Perplexity by Position",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
    rolling_window: int = 10,
) -> Any:
    """
    Plot perplexity as a function of position in sequence.

    Useful for analyzing long-range dependency handling.

    Args:
        position_ppl: Dictionary mapping position to perplexity
        title: Plot title
        figsize: Figure size
        save_path: Path to save
        rolling_window: Window size for smoothing

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()

    positions = sorted(position_ppl.keys())
    ppls = [position_ppl[p] for p in positions]

    fig, ax = plt.subplots(figsize=figsize)

    # Raw values
    ax.plot(positions, ppls, alpha=0.3, color="blue", label="Raw")

    # Rolling average
    if len(ppls) > rolling_window:
        smoothed = np.convolve(ppls, np.ones(rolling_window)/rolling_window, mode="valid")
        smooth_positions = positions[rolling_window//2:rolling_window//2 + len(smoothed)]
        ax.plot(smooth_positions, smoothed, color="blue", linewidth=2,
                label=f"Rolling avg (w={rolling_window})")

    ax.set_xlabel("Position in Sequence")
    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.legend()

    # Add trend annotation
    if len(positions) > 1:
        slope = (ppls[-1] - ppls[0]) / (positions[-1] - positions[0])
        trend = "increasing" if slope > 0 else "decreasing"
        ax.annotate(f"Trend: {trend} ({slope:.4f}/pos)",
                   xy=(0.95, 0.95), xycoords="axes fraction",
                   ha="right", va="top", fontsize=9,
                   bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved perplexity by position to {save_path}")

    return fig


def plot_consolidation_comparison(
    models: Dict[str, List[float]],
    title: str = "Memory Consolidation Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Any:
    """
    Compare consolidation dynamics across different models/configurations.

    Args:
        models: Dictionary mapping model name to list of consolidation R-scores over time
        title: Plot title
        figsize: Figure size
        save_path: Path to save

    Returns:
        matplotlib figure
    """
    plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for (name, r_scores), color in zip(models.items(), colors):
        steps = list(range(len(r_scores)))
        ax.plot(steps, r_scores, label=name, linewidth=2, color=color)
        ax.fill_between(steps, r_scores, alpha=0.1, color=color)

    # Target R-score line
    ax.axhline(y=0.86, color="red", linestyle="--", linewidth=2,
               label="Target R ≥ 0.86")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Consolidation R-Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved consolidation comparison to {save_path}")

    return fig


def create_paper_figures(
    results_dir: str,
    output_dir: str,
    format: str = "pdf",
):
    """
    Generate all figures for paper from saved results.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save figures
        format: Output format (pdf, png, svg)
    """
    import json

    plt = _ensure_matplotlib()
    setup_plotting_style()

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    ablation_path = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            ablation_data = json.load(f)

        # Figure 1: Ablation comparison
        plot_ablation_comparison(
            ablation_data.get("summary", {}),
            metric="perplexity",
            title="Ablation Study: Component Contributions",
            save_path=os.path.join(output_dir, f"fig1_ablation.{format}"),
        )

    # Load training history
    history_path = os.path.join(results_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)

        # Figure 2: Training curves
        plot_training_curves(
            history,
            title="Training Progress",
            save_path=os.path.join(output_dir, f"fig2_training.{format}"),
            smoothing=0.9,
        )

    print(f"Generated paper figures in {output_dir}")
