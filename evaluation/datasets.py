"""
Dataset Loaders for SalienceFormer Evaluation

Support for standard NLP benchmarks and long-context datasets.
"""

import os
from typing import Optional, Dict, Any, List, Callable, Iterator
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

try:
    from datasets import load_dataset, Dataset as HFDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    subset: Optional[str] = None
    split: str = "test"
    max_seq_length: int = 512
    batch_size: int = 4
    num_workers: int = 0
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None

    # For long-context datasets
    stride: int = 256  # Overlap between chunks

    # For QA datasets
    context_field: str = "context"
    question_field: str = "question"
    answer_field: str = "answer"


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset wrapper."""

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        max_samples: Optional[int] = None,
    ):
        self.encodings = encodings
        self.max_samples = max_samples

    def __len__(self) -> int:
        length = len(self.encodings["input_ids"])
        if self.max_samples:
            length = min(length, self.max_samples)
        return length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: val[idx] for key, val in self.encodings.items()}


class StreamingLMDataset(IterableDataset):
    """
    Streaming dataset for large language modeling corpora.

    Handles chunking of long documents with configurable stride.
    """

    def __init__(
        self,
        dataset_name: str,
        subset: Optional[str],
        split: str,
        tokenizer,
        max_seq_length: int,
        stride: int,
        max_samples: Optional[int] = None,
        text_field: str = "text",
    ):
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.stride = stride
        self.max_samples = max_samples
        self.text_field = text_field

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if not HAS_DATASETS:
            raise ImportError("datasets library required")

        dataset = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
        )

        sample_count = 0
        buffer_ids = []

        for example in dataset:
            if self.max_samples and sample_count >= self.max_samples:
                break

            text = example.get(self.text_field, "")
            if not text.strip():
                continue

            # Tokenize and add to buffer
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer_ids.extend(tokens)

            # Yield chunks when buffer is large enough
            while len(buffer_ids) >= self.max_seq_length:
                chunk = buffer_ids[:self.max_seq_length]
                buffer_ids = buffer_ids[self.stride:]  # Sliding window

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(len(chunk), dtype=torch.long),
                }

                sample_count += 1
                if self.max_samples and sample_count >= self.max_samples:
                    break


def load_wikitext103(
    tokenizer,
    config: Optional[DatasetConfig] = None,
) -> DataLoader:
    """
    Load WikiText-103 dataset.

    Larger version of WikiText-2 with ~100M tokens.
    Good for perplexity benchmarking.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required. Install with: pip install datasets")

    config = config or DatasetConfig(name="wikitext", subset="wikitext-103-raw-v1")

    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split=config.split,
        cache_dir=config.cache_dir,
    )

    # Filter empty strings
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if config.max_samples:
        tokenized = tokenized.select(range(min(len(tokenized), config.max_samples)))

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return DataLoader(
        tokenized,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


def load_pg19(
    tokenizer,
    config: Optional[DatasetConfig] = None,
) -> DataLoader:
    """
    Load PG-19 dataset (Project Gutenberg books).

    Long-form text ideal for testing memory over extended contexts.
    Books are chunked with overlap for continuity.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required")

    config = config or DatasetConfig(
        name="pg19",
        max_seq_length=2048,
        stride=512,
    )

    # Use streaming to handle large dataset
    streaming_dataset = StreamingLMDataset(
        dataset_name="pg19",
        subset=None,
        split=config.split,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        stride=config.stride,
        max_samples=config.max_samples,
        text_field="text",
    )

    return DataLoader(
        streaming_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )


def load_narrativeqa(
    tokenizer,
    config: Optional[DatasetConfig] = None,
) -> Dict[str, Any]:
    """
    Load NarrativeQA dataset.

    Question answering over long narratives (books and movie scripts).
    Tests ability to retain and retrieve facts from long context.

    Returns:
        Dictionary with 'dataloader' and 'metadata' (questions, answers, contexts)
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required")

    config = config or DatasetConfig(
        name="narrativeqa",
        max_seq_length=2048,
    )

    dataset = load_dataset(
        "narrativeqa",
        split=config.split,
        cache_dir=config.cache_dir,
    )

    if config.max_samples:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    # Extract components
    contexts = []
    questions = []
    answers = []

    for example in dataset:
        # NarrativeQA has document summaries we can use
        context = example.get("document", {}).get("summary", {}).get("text", "")
        question = example.get("question", {}).get("text", "")
        answer_list = example.get("answers", [])
        answer = answer_list[0].get("text", "") if answer_list else ""

        if context and question:
            contexts.append(context)
            questions.append(question)
            answers.append(answer)

    # Tokenize for model input (context + question)
    def tokenize_qa(idx):
        combined = f"Context: {contexts[idx]}\n\nQuestion: {questions[idx]}\n\nAnswer:"
        return tokenizer(
            combined,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    encodings = {
        "input_ids": [],
        "attention_mask": [],
    }

    for i in range(len(contexts)):
        enc = tokenize_qa(i)
        encodings["input_ids"].append(enc["input_ids"].squeeze(0))
        encodings["attention_mask"].append(enc["attention_mask"].squeeze(0))

    encodings["input_ids"] = torch.stack(encodings["input_ids"])
    encodings["attention_mask"] = torch.stack(encodings["attention_mask"])

    tokenized_dataset = TokenizedDataset(encodings)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return {
        "dataloader": dataloader,
        "contexts": contexts,
        "questions": questions,
        "answers": answers,
    }


def load_scrolls(
    tokenizer,
    subset: str = "qasper",
    config: Optional[DatasetConfig] = None,
) -> Dict[str, Any]:
    """
    Load SCROLLS benchmark dataset.

    Long-context benchmark with multiple tasks:
    - qasper: Scientific paper QA
    - quality: Multiple choice reading comprehension
    - narrative_qa: Story comprehension
    - summ_screen_fd: TV show summarization
    - gov_report: Government report summarization
    - qmsum: Meeting summarization
    - contract_nli: Contract NLI
    """
    if not HAS_DATASETS:
        raise ImportError("datasets library required")

    config = config or DatasetConfig(
        name="scrolls",
        subset=subset,
        max_seq_length=4096,
    )

    dataset = load_dataset(
        "tau/scrolls",
        subset,
        split=config.split,
        cache_dir=config.cache_dir,
    )

    if config.max_samples:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    # SCROLLS has standardized fields
    inputs = []
    targets = []

    for example in dataset:
        inputs.append(example.get("input", ""))
        targets.append(example.get("output", ""))

    def tokenize_fn(text):
        return tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )

    encodings = {
        "input_ids": [],
        "attention_mask": [],
    }

    for inp in inputs:
        enc = tokenize_fn(inp)
        encodings["input_ids"].append(enc["input_ids"].squeeze(0))
        encodings["attention_mask"].append(enc["attention_mask"].squeeze(0))

    encodings["input_ids"] = torch.stack(encodings["input_ids"])
    encodings["attention_mask"] = torch.stack(encodings["attention_mask"])

    tokenized_dataset = TokenizedDataset(encodings)

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return {
        "dataloader": dataloader,
        "inputs": inputs,
        "targets": targets,
    }


def create_eval_dataloader(
    dataset_name: str,
    tokenizer,
    config: Optional[DatasetConfig] = None,
    **kwargs,
) -> DataLoader:
    """
    Factory function to create evaluation dataloaders.

    Args:
        dataset_name: One of 'wikitext-2', 'wikitext-103', 'pg19',
                      'narrativeqa', 'scrolls-*'
        tokenizer: HuggingFace tokenizer
        config: Optional DatasetConfig
        **kwargs: Override config parameters

    Returns:
        DataLoader for the specified dataset
    """
    # Apply kwargs to config
    if config is None:
        config = DatasetConfig(name=dataset_name)

    for k, v in kwargs.items():
        if hasattr(config, k):
            setattr(config, k, v)

    # Route to appropriate loader
    if dataset_name == "wikitext-2":
        if not HAS_DATASETS:
            raise ImportError("datasets library required")

        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split=config.split,
            cache_dir=config.cache_dir,
        )

        dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        if config.max_samples:
            tokenized = tokenized.select(range(min(len(tokenized), config.max_samples)))

        tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

        return DataLoader(
            tokenized,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    elif dataset_name == "wikitext-103":
        return load_wikitext103(tokenizer, config)

    elif dataset_name == "pg19":
        return load_pg19(tokenizer, config)

    elif dataset_name.startswith("scrolls-"):
        subset = dataset_name.replace("scrolls-", "")
        result = load_scrolls(tokenizer, subset, config)
        return result["dataloader"]

    elif dataset_name == "narrativeqa":
        result = load_narrativeqa(tokenizer, config)
        return result["dataloader"]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset.

    Returns:
        Dictionary with dataset metadata
    """
    info = {
        "wikitext-2": {
            "description": "Small Wikipedia-based LM benchmark (~2M tokens)",
            "task": "language_modeling",
            "recommended_seq_length": 512,
            "metrics": ["perplexity"],
        },
        "wikitext-103": {
            "description": "Large Wikipedia-based LM benchmark (~100M tokens)",
            "task": "language_modeling",
            "recommended_seq_length": 512,
            "metrics": ["perplexity"],
        },
        "pg19": {
            "description": "Project Gutenberg books for long-context LM",
            "task": "language_modeling",
            "recommended_seq_length": 2048,
            "metrics": ["perplexity", "perplexity_by_position"],
        },
        "narrativeqa": {
            "description": "QA over books and movie scripts",
            "task": "question_answering",
            "recommended_seq_length": 2048,
            "metrics": ["f1", "rouge_l", "exact_match"],
        },
        "scrolls-qasper": {
            "description": "Scientific paper QA from SCROLLS benchmark",
            "task": "question_answering",
            "recommended_seq_length": 4096,
            "metrics": ["f1", "exact_match"],
        },
        "scrolls-quality": {
            "description": "Multiple choice reading comprehension",
            "task": "multiple_choice",
            "recommended_seq_length": 4096,
            "metrics": ["accuracy"],
        },
    }

    return info.get(dataset_name, {"description": "Unknown dataset"})
