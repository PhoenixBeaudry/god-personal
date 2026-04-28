"""
Baseline stats collection: loss and gradient norms on a training data subset.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from core.models.utility_models import BaselineStats


class SimpleTextDataset(Dataset):
    """Minimal dataset that tokenizes JSON records for loss computation."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 512):
        self.encodings = []
        for record in records:
            text = " ".join(str(v) for v in record.values())
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def compute_baseline_stats(
    model,
    tokenizer,
    data_records: list[dict],
    max_samples: int = 100,
    max_length: int = 512,
) -> BaselineStats:
    """Compute loss and grad norm on a subset of training data."""
    device = next(model.parameters()).device
    records = data_records[:max_samples]

    dataset = SimpleTextDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Forward pass for loss
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # Single forward+backward for grad norm
    model.train()
    model.zero_grad()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    outputs.loss.backward()

    total_grad_norm = 0.0
    n_params = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm(2).item() ** 2
            n_params += 1

    grad_norm = total_grad_norm ** 0.5

    model.eval()
    model.zero_grad()

    return BaselineStats(loss=avg_loss, grad_norm=grad_norm)
