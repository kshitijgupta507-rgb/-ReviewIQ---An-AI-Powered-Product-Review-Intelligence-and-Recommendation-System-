"""DistilBERT fine-tuned sentiment classifier for ReviewIQ.

Uses HuggingFace Transformers:
- Model: ``distilbert-base-uncased`` fine-tuned for sequence classification
- 40% smaller than BERT, 60% faster, retains 97% of BERT's performance
- Automatic hardware acceleration (NVIDIA CUDA GPU if available, else CPU)
- Native PyTorch training loop with AdamW and linear warmup
- Evaluation, prediction, and model checkpoint saving/loading
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from reviewiq.classification.evaluator import evaluate_classifier
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


class _ReviewDataset(Dataset):
    """PyTorch Dataset for tokenized review text."""

    def __init__(
        self,
        encodings: dict[str, torch.Tensor],
        labels: list[int] | None = None,
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


class DistilBertClassifier:
    """DistilBERT sequence classification wrapper.

    Parameters
    ----------
    model_name:
        Pretrained HuggingFace model identifier. Default 'distilbert-base-uncased'.
    num_labels:
        Number of output sentiment classes (default 3: negative, neutral, positive).
    max_length:
        Maximum token sequence length (longer sequences truncated).
    learning_rate:
        Initial learning rate for AdamW optimizer.
    batch_size:
        Batch size for training and inference.
    epochs:
        Number of fine-tuning epochs.
    device:
        Torch device ('cuda', 'cpu', or None for auto-detection).
    """

    def __init__(
        self,
        model_name: str | None = None,
        num_labels: int = 3,
        max_length: int | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config()
        d_cfg = getattr(getattr(cfg, "classification", None), "distilbert", None)

        self.model_name = (
            model_name
            if model_name is not None
            else getattr(d_cfg, "model_name", "distilbert-base-uncased")
        )
        self.num_labels = num_labels
        self.max_length = (
            max_length if max_length is not None else getattr(d_cfg, "max_length", 256)
        )
        self.learning_rate = (
            learning_rate
            if learning_rate is not None
            else getattr(d_cfg, "learning_rate", 2e-5)
        )
        self.batch_size = (
            batch_size if batch_size is not None else getattr(d_cfg, "batch_size", 16)
        )
        self.epochs = (
            epochs if epochs is not None else getattr(d_cfg, "num_epochs", 3)
        )

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer: AutoTokenizer | None = None
        self.model: AutoModelForSequenceClassification | None = None
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}
        self.is_fitted = False

        logger.info(
            "DistilBERT initialized on device: %s (model: %s).",
            self.device,
            self.model_name,
        )

    def _init_model(self, labels: list[str]) -> None:
        """Initialize HuggingFace tokenizer and classification head."""
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.id_to_label = {i: label for i, label in enumerate(labels)}
        self.num_labels = len(labels)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        ).to(self.device)

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        val_texts: Sequence[str] | None = None,
        val_labels: Sequence[str] | None = None,
    ) -> "DistilBertClassifier":
        """Fine-tune DistilBERT on text reviews and sentiment labels."""
        unique_labels = sorted(list(set(labels)))
        self._init_model(unique_labels)

        train_label_ids = [self.label_to_id[lbl] for lbl in labels]

        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _ReviewDataset(encodings, train_label_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(dataloader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                b_labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=b_labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            logger.info("Epoch %d/%d completed. Training loss: %.4f", epoch + 1, self.epochs, avg_loss)

        self.is_fitted = True
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Predict class probability distribution for input strings."""
        if not self.is_fitted or self.model is None or self.tokenizer is None:
            raise ValueError("DistilBertClassifier must be fitted or loaded before predicting.")

        self.model.eval()
        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dataset = _ReviewDataset(encodings)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts: Sequence[str]) -> list[str]:
        """Predict sentiment string labels."""
        probs = self.predict_proba(texts)
        pred_ids = np.argmax(probs, axis=1)
        return [self.id_to_label[i] for i in pred_ids]

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> dict[str, Any]:
        """Evaluate model performance on test texts and labels."""
        preds = self.predict(texts)
        return evaluate_classifier(
            labels,
            preds,
            model_name="DistilBERT",
            representation_name="Transformer Fine-Tuned",
            labels=sorted(list(self.label_to_id.keys())),
        )

    def save(self, output_dir: str | Path) -> None:
        """Save fine-tuned model and tokenizer artifacts."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.model is not None and self.tokenizer is not None:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            logger.info("Saved DistilBERT model & tokenizer to %s.", path)

    @classmethod
    def load(cls, model_dir: str | Path, device: str | None = None) -> "DistilBertClassifier":
        """Load fine-tuned model and tokenizer from disk."""
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model directory not found at {path}")

        instance = cls(device=device)
        instance.tokenizer = AutoTokenizer.from_pretrained(str(path))
        instance.model = AutoModelForSequenceClassification.from_pretrained(str(path)).to(instance.device)
        raw_id2label = instance.model.config.id2label or {}
        instance.id_to_label = {int(k): v for k, v in raw_id2label.items()}
        instance.label_to_id = instance.model.config.label2id or {}
        instance.num_labels = len(instance.id_to_label)
        instance.is_fitted = True
        logger.info("Loaded DistilBERT model from %s on device: %s.", path, instance.device)
        return instance
