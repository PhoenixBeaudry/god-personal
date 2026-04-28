"""
Comprehensive stats collection: dataset analysis, weight structure, and training dynamics.
All computed in a single pass inside the model prep container.
"""

import math
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.model_prep_models import (
    BaselineStats,
    DatasetStats,
    LayerGradStats,
    LayerGroupWeightStats,
    SeqLengthDistribution,
    TrainingDynamics,
    WeightStats,
)

BPB_REFERENCE_MODEL = "gpt2"


# --- Tokenization helper ---

class SimpleTextDataset(Dataset):
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


# --- Layer type classification ---

LAYER_TYPE_PATTERNS = {
    "attention_qkv": [
        r"\.q_proj\.", r"\.k_proj\.", r"\.v_proj\.", r"\.c_attn\.",
        r"\.query\.", r"\.key\.", r"\.value\.", r"\.qkv\.", r"\.Wqkv\.",
        r"\.query_key_value\.",
    ],
    "attention_output": [
        r"\.o_proj\.", r"\.out_proj\.", r"\.attn\.c_proj\.", r"\.dense\b(?=.*attn)",
        r"\.self_attn\.dense\.",
    ],
    "ffn_up": [
        r"\.up_proj\.", r"\.gate_proj\.", r"\.c_fc\.", r"\.fc1\.",
        r"\.wi\.", r"\.dense_h_to_4h\.", r"\.gate\.",
        r"\.w1\.", r"\.w3\.",
    ],
    "ffn_down": [
        r"\.down_proj\.", r"\.mlp\.c_proj\.", r"\.fc2\.",
        r"\.wo\.", r"\.dense_4h_to_h\.", r"\.w2\.",
    ],
    "embedding": [
        r"embed_tokens\.", r"\.wte\.", r"word_embeddings\.", r"embed_in\.",
    ],
    "unembedding": [
        r"lm_head\.", r"embed_out\.",
    ],
    "layer_norm": [
        r"layernorm", r"layer_norm", r"\.ln_", r"\.norm\.", r"rmsnorm",
        r"\.ln_f\.", r"final_layer_norm",
    ],
}


def classify_layer(name: str) -> str:
    name_lower = name.lower()
    for group, patterns in LAYER_TYPE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, name_lower):
                return group
    return "other"


# --- Dataset stats ---

def _compute_dataset_stats(records: list[dict], tokenizer, device: str = "cpu") -> DatasetStats:
    texts = [" ".join(str(v) for v in r.values()) for r in records]
    token_lengths = []
    for text in texts:
        tokens = tokenizer(text, truncation=False)
        token_lengths.append(len(tokens["input_ids"]))

    lengths_arr = np.array(token_lengths)

    seq_dist = SeqLengthDistribution(
        mean=float(np.mean(lengths_arr)),
        p50=int(np.percentile(lengths_arr, 50)),
        p95=int(np.percentile(lengths_arr, 95)),
        p99=int(np.percentile(lengths_arr, 99)),
        max=int(np.max(lengths_arr)),
    )

    dup_rate = _compute_near_duplicate_rate(texts)
    bpb = _compute_bits_per_byte(texts, device)

    return DatasetStats(
        total_tokens=int(np.sum(lengths_arr)),
        seq_length_distribution=seq_dist,
        near_duplicate_rate=dup_rate,
        bits_per_byte=bpb,
        vocab_size=len(tokenizer),
    )


def _compute_near_duplicate_rate(texts: list[str], num_perm: int = 128, threshold: float = 0.5) -> float:
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes = []
        for i, text in enumerate(texts):
            m = MinHash(num_perm=num_perm)
            for word in text.lower().split():
                m.update(word.encode("utf-8"))
            minhashes.append(m)
            try:
                lsh.insert(str(i), m)
            except ValueError:
                pass  # duplicate key

        dup_count = 0
        for i, m in enumerate(minhashes):
            results = lsh.query(m)
            if len(results) > 1:
                dup_count += 1

        return dup_count / max(len(texts), 1)
    except ImportError:
        print("Warning: datasketch not installed, skipping near-duplicate rate")
        return 0.0


def _compute_bits_per_byte(texts: list[str], device: str = "cpu") -> float:
    ref_model = AutoModelForCausalLM.from_pretrained(BPB_REFERENCE_MODEL).to(device)
    ref_tokenizer = AutoTokenizer.from_pretrained(BPB_REFERENCE_MODEL)
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_model.eval()

    total_loss_nats = 0.0
    total_bytes = 0

    with torch.no_grad():
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            enc = ref_tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)
            outputs = ref_model(**enc, labels=enc["input_ids"])
            n_tokens = enc["input_ids"].shape[1]
            total_loss_nats += outputs.loss.item() * n_tokens

    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if total_bytes == 0:
        return 0.0
    return (total_loss_nats / math.log(2)) / total_bytes


# --- Weight stats ---

def _compute_weight_stats(model) -> WeightStats:
    group_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)

    for name, param in model.named_parameters():
        group = classify_layer(name)
        group_tensors[group].append(param.data.float())

    by_group = {}
    for group, tensors in group_tensors.items():
        all_weights = torch.cat([t.flatten() for t in tensors])
        by_group[group] = LayerGroupWeightStats(
            weight_rms=float(torch.sqrt(torch.mean(all_weights ** 2)).item()),
            weight_norm=float(torch.norm(all_weights).item()),
            max_abs=float(torch.max(torch.abs(all_weights)).item()),
        )

    return WeightStats(by_group=by_group)


# --- Training dynamics ---

def _get_model_device(model) -> torch.device:
    """Get the device for input tensors. Handles device_map='auto' (multi-GPU)."""
    if hasattr(model, "hf_device_map"):
        return torch.device("cuda:0")
    return next(model.parameters()).device


def _compute_training_dynamics(
    model,
    tokenizer,
    records: list[dict],
    n_samples: int = 100,
    n_subbatches: int = 10,
    max_length: int = 512,
) -> TrainingDynamics:
    device = _get_model_device(model)
    records = records[:n_samples]

    dataset = SimpleTextDataset(records, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- Init loss + output entropy ---
    model.eval()
    total_loss = 0.0
    total_entropy = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()

            probs = F.softmax(outputs.logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1).mean().item()
            total_entropy += entropy
            n_batches += 1

    init_loss = total_loss / max(n_batches, 1)
    output_entropy = total_entropy / max(n_batches, 1)

    # --- Forward hooks for activation RMS ---
    activation_rms_accum: dict[str, list[float]] = defaultdict(list)
    hooks = []

    def make_activation_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            if isinstance(out, torch.Tensor):
                rms = torch.sqrt(torch.mean(out.float() ** 2)).item()
                activation_rms_accum[name].append(rms)
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters(recurse=False)):
            hooks.append(module.register_forward_hook(make_activation_hook(name)))

    # --- Backward hooks for grad stats ---
    grad_accum: dict[str, list[torch.Tensor]] = defaultdict(list)

    def make_grad_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                grad_accum[name].append(grad_output[0].detach().float())
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters(recurse=False)):
            hooks.append(module.register_full_backward_hook(make_grad_hook(name)))

    # --- Single forward+backward for hooks ---
    model.train()
    model.zero_grad()
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    outputs.loss.backward()

    # Per-layer grad norms
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = float(param.grad.norm(2).item())

    # Per-layer grad stats with SVD
    grad_stats = {}
    for name, grads in grad_accum.items():
        if not grads:
            continue
        g = grads[0]
        if g.dim() < 2:
            g = g.unsqueeze(0)
        g_2d = g.reshape(g.shape[0], -1) if g.dim() > 2 else g

        frob = float(torch.norm(g_2d).item())
        rms = float(torch.sqrt(torch.mean(g_2d ** 2)).item())
        max_abs = float(torch.max(torch.abs(g_2d)).item())

        k = min(64, min(g_2d.shape))
        try:
            _, s, _ = torch.svd_lowrank(g_2d.float(), q=k)
            top_sv = s.tolist()
        except Exception:
            top_sv = []

        grad_stats[name] = LayerGradStats(
            frobenius_norm=frob,
            rms=rms,
            max_abs=max_abs,
            top_singular_values=top_sv,
        )

    # Remove hooks
    for h in hooks:
        h.remove()

    # Activation RMS averages
    activation_rms = {name: float(np.mean(vals)) for name, vals in activation_rms_accum.items()}

    # --- Gradient noise scale ---
    gradient_noise_scale = _compute_gradient_noise_scale(model, loader, device, n_subbatches)

    model.eval()
    model.zero_grad()

    return TrainingDynamics(
        init_loss=init_loss,
        grad_norms=grad_norms,
        gradient_noise_scale=gradient_noise_scale,
        activation_rms=activation_rms,
        grad_stats=grad_stats,
        output_entropy=output_entropy,
    )


def _compute_gradient_noise_scale(model, loader, device, n_subbatches: int) -> float:
    """Approximate gradient noise scale: B_noise ≈ Var(g) / ||E[g]||^2"""
    all_batches = list(loader)
    if len(all_batches) < n_subbatches:
        return 0.0

    chunk_size = len(all_batches) // n_subbatches
    subbatch_grads = []

    for i in range(n_subbatches):
        chunk = all_batches[i * chunk_size:(i + 1) * chunk_size]
        model.zero_grad()

        total_loss = 0.0
        for batch in chunk:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss

        (total_loss / len(chunk)).backward()

        flat_grad = torch.cat([
            p.grad.flatten() for p in model.parameters() if p.grad is not None
        ])
        subbatch_grads.append(flat_grad.detach())

    grad_stack = torch.stack(subbatch_grads)
    mean_grad = grad_stack.mean(dim=0)
    var_grad = grad_stack.var(dim=0).sum().item()
    mean_norm_sq = mean_grad.norm(2).item() ** 2

    if mean_norm_sq < 1e-12:
        return 0.0

    return var_grad / mean_norm_sq


# --- Main entry point ---

def compute_baseline_stats(
    model,
    tokenizer,
    data_records: list[dict],
    max_samples: int = 100,
) -> BaselineStats:
    device = str(_get_model_device(model))

    print("Computing dataset stats...", flush=True)
    dataset_stats = _compute_dataset_stats(data_records[:max_samples], tokenizer, device)

    print("Computing weight stats...", flush=True)
    weight_stats = _compute_weight_stats(model)

    print("Computing training dynamics...", flush=True)
    training_dynamics = _compute_training_dynamics(model, tokenizer, data_records, n_samples=max_samples)

    return BaselineStats(
        dataset=dataset_stats,
        weights=weight_stats,
        training=training_dynamics,
    )
