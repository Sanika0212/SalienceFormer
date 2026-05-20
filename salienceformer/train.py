"""
SalienceFormer Training Script

Training loop for SalienceFormer with:
- HuggingFace datasets integration
- Gradient accumulation
- Mixed precision training
- Logging and checkpointing
"""

import os
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from salienceformer.config import SalienceFormerConfig
from salienceformer.model import SalienceFormer


@dataclass
class TrainingArgs:
    """Training arguments."""

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_seq_length: int = 512
    batch_size: int = 4
    gradient_accumulation_steps: int = 4

    # Training
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"

    # Logging & Checkpoints
    output_dir: str = "./salienceformer_output"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "salienceformer"
    wandb_run_name: Optional[str] = None

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None


class SalienceFormerTrainer:
    """Trainer for SalienceFormer."""

    def __init__(
        self,
        model: SalienceFormer,
        args: TrainingArgs,
        tokenizer: Optional[Any] = None,
    ):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup AMP
        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        self.amp_dtype = getattr(torch, args.amp_dtype) if args.use_amp else torch.float32

        # Tracking
        self.global_step = 0
        self.epoch = 0

        # Initialize W&B
        self.use_wandb = args.use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "num_epochs": args.num_epochs,
                    "max_seq_length": args.max_seq_length,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio,
                    "use_amp": args.use_amp,
                    "total_params": model.get_num_total_params(),
                    "trainable_params": model.get_num_trainable_params(),
                },
            )
            wandb.watch(model, log="gradients", log_freq=100)

        # Resume from checkpoint if specified
        if args.resume_from_checkpoint:
            self._load_checkpoint(args.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        ckpt_file = os.path.join(checkpoint_path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            print(f"Warning: Checkpoint not found at {ckpt_file}, starting fresh")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        print(f"Resumed from epoch {self.epoch}, step {self.global_step}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for different components."""
        # Separate parameters by component
        base_params = []
        hippo_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'base_model' in name:
                    base_params.append(param)
                else:
                    hippo_params.append(param)

        param_groups = [
            {"params": hippo_params, "lr": self.args.learning_rate},
            {"params": base_params, "lr": self.args.learning_rate * 0.1},  # Lower LR for base
        ]

        return AdamW(
            param_groups,
            weight_decay=self.args.weight_decay,
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.args.warmup_ratio)
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation

        Returns:
            Training history dictionary
        """
        num_training_steps = (
            len(train_dataloader) * self.args.num_epochs //
            self.args.gradient_accumulation_steps
        )
        scheduler = self._create_scheduler(num_training_steps)

        # Fast-forward scheduler if resuming
        starting_step = self.global_step
        if starting_step > 0:
            print(f"Resuming from step {starting_step}, fast-forwarding scheduler...")
            for _ in range(starting_step):
                scheduler.step()

        # Store starting step for _train_epoch to use
        self._starting_step = starting_step

        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)

        history = {
            "train_loss": [],
            "eval_loss": [],
            "salience_stats": [],
            "memory_stats": [],
        }

        print(f"Starting training for {self.args.num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"Trainable parameters: {self.model.get_num_trainable_params():,}")

        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch(
                train_dataloader,
                scheduler,
                history,
            )

            print(f"Epoch {epoch + 1}/{self.args.num_epochs}, Loss: {epoch_loss:.4f}")

            # Evaluation
            if eval_dataloader is not None:
                eval_loss = self._evaluate(eval_dataloader)
                history["eval_loss"].append(eval_loss)
                print(f"Eval Loss: {eval_loss:.4f}")
                if self.use_wandb:
                    wandb.log({"eval/loss": eval_loss, "eval/epoch": epoch + 1})

            # Save checkpoint
            self._save_checkpoint(epoch)

        # Finish W&B run
        if self.use_wandb:
            wandb.finish()

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        scheduler,
        history: Dict,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        accumulated_loss = 0.0

        # Calculate step to resume from (only applies once at start)
        resume_step = getattr(self, '_starting_step', 0) * self.args.gradient_accumulation_steps

        for step, batch in enumerate(dataloader):
            # Skip steps already processed (for resuming)
            if step < resume_step:
                if step % 5000 == 0:
                    print(f"Skipping to step {step}/{resume_step}...")
                continue
            elif step == resume_step and resume_step > 0:
                print(f"Resuming training from batch {step}")
                self._starting_step = 0  # Clear so subsequent epochs don't skip
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=self.args.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_salience=True,
                    return_memory_stats=(step % self.args.logging_steps == 0),
                )
                loss = outputs["loss"] / self.args.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation step
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm,
                )

                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                history["train_loss"].append(accumulated_loss)

                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    self._log_step(
                        step=self.global_step,
                        loss=accumulated_loss,
                        outputs=outputs,
                        lr=current_lr,
                    )

                accumulated_loss = 0.0

                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self._save_checkpoint(self.epoch, is_step=True)

        return total_loss / (len(dataloader) // self.args.gradient_accumulation_steps)

    def _evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on dataloader."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                labels = batch.get("labels", input_ids).to(self.device)

                with torch.cuda.amp.autocast(enabled=self.args.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    total_loss += outputs["loss"].item()

        return total_loss / len(dataloader)

    def _log_step(
        self,
        step: int,
        loss: float,
        outputs: Dict,
        lr: Optional[float] = None,
    ) -> None:
        """Log training step."""
        log_str = f"Step {step}, Loss: {loss:.4f}"
        wandb_log = {"train/loss": loss, "train/step": step}

        if lr is not None:
            wandb_log["train/learning_rate"] = lr

        if "salience_stats" in outputs:
            stats = outputs["salience_stats"]
            log_str += f", Salience: {stats['mean_salience']:.3f} (tagged: {stats['tagged_ratio']:.2%})"
            wandb_log["salience/mean"] = stats["mean_salience"]
            wandb_log["salience/tagged_ratio"] = stats["tagged_ratio"]

        if "memory_stats" in outputs:
            stats = outputs["memory_stats"]
            log_str += f", Buffer: {stats['buffer_utilization']:.1%}"
            wandb_log["memory/buffer_utilization"] = stats["buffer_utilization"]

        print(log_str)

        if self.use_wandb:
            wandb.log(wandb_log, step=step)

    def _save_checkpoint(self, epoch: int, is_step: bool = False) -> None:
        """Save model checkpoint."""
        if is_step:
            path = os.path.join(self.args.output_dir, f"checkpoint-step-{self.global_step}")
        else:
            path = os.path.join(self.args.output_dir, f"checkpoint-epoch-{epoch}")

        os.makedirs(path, exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
            },
            os.path.join(path, "checkpoint.pt"),
        )

        # Save config
        torch.save(self.model.config, os.path.join(path, "config.pt"))

        print(f"Saved checkpoint to {path}")


def create_dataloaders(
    tokenizer,
    args: TrainingArgs,
) -> tuple:
    """Create train and eval dataloaders from HuggingFace datasets."""
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")

    # Dataset-specific loading
    DATASET_CONFIGS = {
        "wikitext": {
            "text_field": "text",
            "streaming": False,
        },
        "togethercomputer/RedPajama-Data-1T-Sample": {
            "text_field": "text",
            "streaming": True,  # Large dataset, use streaming
        },
        "openwebtext": {
            "text_field": "text",
            "streaming": True,
        },
        "EleutherAI/pile": {
            "text_field": "text",
            "streaming": True,
        },
    }

    dataset_info = DATASET_CONFIGS.get(args.dataset_name, {"text_field": "text", "streaming": False})
    text_field = dataset_info["text_field"]
    use_streaming = dataset_info.get("streaming", False)

    print(f"Loading dataset: {args.dataset_name}")
    print(f"Config: {args.dataset_config}")
    print(f"Streaming: {use_streaming}")

    if use_streaming:
        # Streaming mode for large datasets
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config if args.dataset_config else None,
            split="train",
            streaming=True,
        )

        def tokenize_function(examples):
            texts = examples[text_field] if isinstance(examples[text_field], list) else [examples[text_field]]
            texts = [t for t in texts if t and len(t.strip()) > 0]
            if not texts:
                return {"input_ids": [], "attention_mask": []}
            return tokenizer(
                texts,
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=[text_field])

        # For streaming, we use IterableDataset
        from torch.utils.data import IterableDataset

        class StreamingDataset(IterableDataset):
            def __init__(self, dataset, max_samples=None):
                self.dataset = dataset
                self.max_samples = max_samples

            def __iter__(self):
                count = 0
                for item in self.dataset:
                    if self.max_samples and count >= self.max_samples:
                        break
                    if "input_ids" in item and len(item["input_ids"]) > 0:
                        yield {
                            "input_ids": torch.tensor(item["input_ids"]),
                            "attention_mask": torch.tensor(item["attention_mask"]),
                        }
                        count += 1

        train_dataset = StreamingDataset(tokenized, max_samples=getattr(args, 'max_samples', None))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

        # For eval, use WikiText-2 validation (standard benchmark)
        print("Loading WikiText-2 validation for evaluation...")
        eval_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        eval_dataset = eval_dataset.filter(lambda x: len(x["text"].strip()) > 0)

        def eval_tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )

        eval_tokenized = eval_dataset.map(eval_tokenize, batched=True, remove_columns=["text"])
        eval_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
        eval_dataloader = DataLoader(eval_tokenized, batch_size=args.batch_size, shuffle=False)

    else:
        # Standard mode for smaller datasets
        dataset = load_dataset(args.dataset_name, args.dataset_config)

        def tokenize_function(examples):
            return tokenizer(
                examples[text_field],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
            )

        # Tokenize
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Set format for PyTorch
        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized["train"],
            batch_size=args.batch_size,
            shuffle=True,
        )

        eval_dataloader = DataLoader(
            tokenized["validation"] if "validation" in tokenized else tokenized["test"],
            batch_size=args.batch_size,
            shuffle=False,
        )

    return train_dataloader, eval_dataloader


def main():
    """Main training entry point with CLI support."""
    import argparse

    parser = argparse.ArgumentParser(description="Train SalienceFormer")

    # Model arguments
    parser.add_argument("--base_model", type=str, default="google/gemma-2b",
                        help="Base model (google/gemma-2b, meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")

    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "togethercomputer/RedPajama-Data-1T-Sample", "openwebtext"],
                        help="Dataset to use")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset config (e.g., wikitext-2-raw-v1, wikitext-103-raw-v1)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples for streaming datasets")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps (overrides epochs)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./salienceformer_output",
                        help="Output directory")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")

    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="salienceformer", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")

    # Resume
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint path")

    cli_args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("SalienceFormer Training Configuration")
    print("=" * 60)
    print(f"Base model:     {cli_args.base_model}")
    print(f"Dataset:        {cli_args.dataset} ({cli_args.dataset_config})")
    print(f"Batch size:     {cli_args.batch_size} x {cli_args.gradient_accumulation_steps} = {cli_args.batch_size * cli_args.gradient_accumulation_steps}")
    print(f"Sequence length: {cli_args.max_seq_length}")
    print(f"Learning rate:  {cli_args.learning_rate}")
    print(f"Output:         {cli_args.output_dir}")
    print("=" * 60)

    # Model configuration
    config = SalienceFormerConfig(
        base_model_name=cli_args.base_model,
        freeze_base=True,
        use_lora=True,
        lora_r=cli_args.lora_r,
        lora_alpha=cli_args.lora_alpha,
    )

    # Training arguments
    args = TrainingArgs(
        dataset_name=cli_args.dataset,
        dataset_config=cli_args.dataset_config if cli_args.dataset == "wikitext" else None,
        batch_size=cli_args.batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        max_seq_length=cli_args.max_seq_length,
        num_epochs=cli_args.num_epochs,
        learning_rate=cli_args.learning_rate,
        warmup_ratio=cli_args.warmup_ratio,
        max_grad_norm=0.5,
        use_amp=True,
        amp_dtype="bfloat16",
        output_dir=cli_args.output_dir,
        save_steps=cli_args.save_steps,
        logging_steps=cli_args.logging_steps,
        use_wandb=not cli_args.no_wandb,
        wandb_project=cli_args.wandb_project,
        wandb_run_name=cli_args.wandb_run_name or f"salienceformer-{cli_args.base_model.split('/')[-1]}-{cli_args.dataset.split('/')[-1]}",
        resume_from_checkpoint=cli_args.resume_from_checkpoint,
    )

    # Add max_samples to args if provided
    if cli_args.max_samples:
        args.max_samples = cli_args.max_samples

    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Creating SalienceFormer model...")
    model = SalienceFormer(config)

    print(f"Total parameters: {model.get_num_total_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")

    print("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)

    print("Starting training...")
    trainer = SalienceFormerTrainer(model, args, tokenizer)
    history = trainer.train(train_dataloader, eval_dataloader)

    print("Training complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['eval_loss']:
        print(f"Final eval loss: {history['eval_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
