"""
Comprehensive stats collection: dataset analysis, weight structure, and training dynamics.
Per-type stats for instruct, DPO, GRPO, and chat tasks.
"""

import math
import re
from collections import defaultdict
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.models.model_prep_models import (
    BaselineStats,
    DatasetStatsBase,
    DpoBaselineStats,
    DpoDatasetStats,
    DpoTrainingDynamics,
    GrpoBaselineStats,
    GrpoDatasetStats,
    GrpoTrainingDynamics,
    InstructBaselineStats,
    InstructDatasetStats,
    InstructTrainingDynamics,
    LayerGradStats,
    LayerGroupWeightStats,
    SeqLengthDistribution,
    TrainingDynamicsBase,
    WeightStats,
)

BPB_REFERENCE_MODEL = "gpt2"


# --- Text extraction per task type ---

def _extract_instruct_texts(records: list[dict]) -> list[tuple[str, str]]:
    """Returns list of (prompt, completion) tuples."""
    results = []
    for r in records:
        prompt_parts = []
        if r.get("system"):
            prompt_parts.append(str(r["system"]))
        if r.get("instruct"):
            prompt_parts.append(str(r["instruct"]))
        if r.get("input"):
            prompt_parts.append(str(r["input"]))
        prompt = " ".join(prompt_parts) or " ".join(str(v) for v in r.values())
        completion = str(r.get("output", ""))
        results.append((prompt, completion))
    return results


def _extract_dpo_texts(records: list[dict]) -> list[tuple[str, str, str]]:
    """Returns list of (prompt, chosen, rejected) tuples."""
    results = []
    for r in records:
        prompt = str(r.get("prompt", ""))
        chosen = str(r.get("chosen", ""))
        rejected = str(r.get("rejected", ""))
        results.append((prompt, chosen, rejected))
    return results


def _extract_grpo_texts(records: list[dict]) -> list[str]:
    """Returns list of prompt strings."""
    return [str(r.get("prompt", " ".join(str(v) for v in r.values()))) for r in records]


def _extract_chat_texts(records: list[dict]) -> list[tuple[str, str]]:
    """Returns list of (prompt, completion) tuples from conversation turns."""
    results = []
    for r in records:
        convos = r.get("conversations", [])
        if isinstance(convos, str):
            results.append((convos, ""))
            continue
        user_parts = []
        assistant_parts = []
        for turn in convos:
            role = str(turn.get("from", turn.get("role", "")))
            content = str(turn.get("value", turn.get("content", "")))
            if role in ("user", "human"):
                user_parts.append(content)
            elif role in ("assistant", "gpt", "bot"):
                assistant_parts.append(content)
        results.append((" ".join(user_parts), " ".join(assistant_parts)))
    return results


# --- Tokenization helper ---

class SimpleTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        self.encodings = []
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


# --- Seq length distribution helper ---

def _make_seq_dist(lengths: list[int]) -> SeqLengthDistribution:
    arr = np.array(lengths) if lengths else np.array([0])
    return SeqLengthDistribution(
        mean=float(np.mean(arr)),
        p50=int(np.percentile(arr, 50)),
        p95=int(np.percentile(arr, 95)),
        p99=int(np.percentile(arr, 99)),
        max=int(np.max(arr)),
    )


def _token_lengths(texts: list[str], tokenizer) -> list[int]:
    return [len(tokenizer(t, truncation=False)["input_ids"]) for t in texts]


# --- Layer type classification ---

LAYER_TYPE_PATTERNS = {
    "attention_qkv": [
        r"\.q_proj\.", r"\.k_proj\.", r"\.v_proj\.", r"\.c_attn\.",
        r"\.query\.", r"\.key\.", r"\.value\.", r"\.qkv\.", r"\.Wqkv\.",
        r"\.query_key_value\.",
    ],
    "attention_output": [
        r"\.o_proj\.", r"\.out_proj\.", r"\.attn\.c_proj\.",
        r"self_attn.*\.dense\.", r"self_attention.*\.dense\.",
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


# --- Shared computations ---

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
                pass
        dup_count = sum(1 for i, m in enumerate(minhashes) if len(lsh.query(m)) > 1)
        return dup_count / max(len(texts), 1)
    except ImportError:
        return 0.0


def _compute_bits_per_byte(texts: list[str], device: str = "cpu") -> float:
    texts = [t for t in texts if t.strip()]
    if not texts:
        return 0.0
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
            total_loss_nats += outputs.loss.item() * enc["input_ids"].shape[1]
    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (total_loss_nats / math.log(2)) / max(total_bytes, 1)


def _compute_weight_stats(model) -> WeightStats:
    group_tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, param in model.named_parameters():
        group_tensors[classify_layer(name)].append(param.data.float())
    by_group = {}
    for group, tensors in group_tensors.items():
        all_w = torch.cat([t.flatten() for t in tensors])
        by_group[group] = LayerGroupWeightStats(
            weight_rms=float(torch.sqrt(torch.mean(all_w ** 2)).item()),
            weight_norm=float(torch.norm(all_w).item()),
            max_abs=float(torch.max(torch.abs(all_w)).item()),
        )
    return WeightStats(by_group=by_group)


def _get_model_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        return torch.device("cuda:0")
    return next(model.parameters()).device


def _compute_base_training_dynamics(
    model, tokenizer, texts: list[str], device, n_subbatches: int = 10, max_length: int = 512,
) -> dict:
    """Compute shared training dynamics (loss, grads, activations, SVD, entropy, noise scale)."""
    dataset = SimpleTextDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Init loss + entropy
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
            total_entropy += -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
            n_batches += 1

    init_loss = total_loss / max(n_batches, 1)
    output_entropy = total_entropy / max(n_batches, 1)

    # Forward hooks for activation RMS
    activation_rms_accum: dict[str, list[float]] = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            if isinstance(out, torch.Tensor):
                activation_rms_accum[name].append(torch.sqrt(torch.mean(out.float() ** 2)).item())
        return hook

    for name, module in model.named_modules():
        if not list(module.children()) and any(p.requires_grad for p in module.parameters(recurse=False)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward+backward for grads
    model.train()
    model.zero_grad()
    batch = next(iter(loader))
    outputs = model(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["input_ids"].to(device),
    )
    outputs.loss.backward()

    grad_norms = {}
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad_norms[name] = float(param.grad.norm(2).item())
        g = param.grad.detach().float()
        if g.dim() < 2:
            g = g.unsqueeze(0)
        g_2d = g.reshape(g.shape[0], -1) if g.dim() > 2 else g
        k = min(8, min(g_2d.shape))
        try:
            _, s, _ = torch.svd_lowrank(g_2d, q=k)
            top_sv = s.tolist()
        except Exception:
            top_sv = []
        grad_stats[name] = LayerGradStats(
            frobenius_norm=float(torch.norm(g_2d).item()),
            rms=float(torch.sqrt(torch.mean(g_2d ** 2)).item()),
            max_abs=float(torch.max(torch.abs(g_2d)).item()),
            top_singular_values=top_sv,
        )

    for h in hooks:
        h.remove()

    activation_rms = {n: float(np.mean(v)) for n, v in activation_rms_accum.items()}
    noise_scale = _compute_gradient_noise_scale(model, loader, device, n_subbatches)

    model.eval()
    model.zero_grad()

    return {
        "init_loss": init_loss,
        "grad_norms": grad_norms,
        "gradient_noise_scale": noise_scale,
        "activation_rms": activation_rms,
        "grad_stats": grad_stats,
        "output_entropy": output_entropy,
    }


def _compute_gradient_noise_scale(model, loader, device, n_subbatches: int) -> float:
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
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["input_ids"].to(device),
            )
            total_loss += outputs.loss
        (total_loss / len(chunk)).backward()
        flat_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        subbatch_grads.append(flat_grad.detach())
    grad_stack = torch.stack(subbatch_grads)
    mean_norm_sq = grad_stack.mean(dim=0).norm(2).item() ** 2
    if mean_norm_sq < 1e-12:
        return 0.0
    return grad_stack.var(dim=0).sum().item() / mean_norm_sq


def _compute_masked_loss(model, tokenizer, prompt_texts: list[str], completion_texts: list[str], device, max_length: int = 512) -> float:
    """Compute loss masked to completion tokens only."""
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for prompt, completion in zip(prompt_texts, completion_texts):
            if not completion.strip():
                continue
            full_text = prompt + " " + completion
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = full_enc["input_ids"].clone()
            labels[0, :prompt_len] = -100  # mask prompt tokens

            outputs = model(**full_enc, labels=labels)
            total_loss += outputs.loss.item()
            n += 1
    return total_loss / max(n, 1)


def _compute_log_probs(model, tokenizer, prompts: list[str], completions: list[str], device, max_length: int = 512) -> list[float]:
    """Compute mean log-prob of completions given prompts."""
    model.eval()
    log_probs = []
    with torch.no_grad():
        for prompt, completion in zip(prompts, completions):
            if not completion.strip():
                continue
            full_text = prompt + " " + completion
            full_enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            outputs = model(**full_enc)
            logits = outputs.logits[0]  # (seq_len, vocab)

            # Get log-probs for completion tokens
            completion_logits = logits[prompt_len - 1:-1]  # shifted
            completion_targets = full_enc["input_ids"][0, prompt_len:]
            if completion_targets.shape[0] == 0:
                continue

            log_p = F.log_softmax(completion_logits, dim=-1)
            token_log_probs = log_p.gather(1, completion_targets.unsqueeze(1)).squeeze(1)
            log_probs.append(token_log_probs.mean().item())
    return log_probs


# --- Per-type compute functions ---

def _compute_instruct_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
    text_extractor=None,
) -> InstructBaselineStats:
    extractor = text_extractor or _extract_instruct_texts
    texts = extractor(records[:max_samples])
    prompts, completions = zip(*texts) if texts else ([], [])
    all_texts = [p + " " + c for p, c in texts]

    prompt_lengths = _token_lengths(list(prompts), tokenizer)
    completion_lengths = _token_lengths(list(completions), tokenizer)

    dataset_stats = InstructDatasetStats(
        total_tokens=sum(prompt_lengths) + sum(completion_lengths),
        seq_length_distribution=_make_seq_dist([p + c for p, c in zip(prompt_lengths, completion_lengths)]),
        near_duplicate_rate=_compute_near_duplicate_rate(all_texts),
        bits_per_byte=_compute_bits_per_byte(list(completions), device),
        vocab_size=len(tokenizer),
        prompt_tokens=sum(prompt_lengths),
        completion_tokens=sum(completion_lengths),
        completion_length_distribution=_make_seq_dist(completion_lengths),
    )

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, all_texts, device)
    masked_loss = _compute_masked_loss(model, tokenizer, list(prompts), list(completions), device)

    training = InstructTrainingDynamics(**base_dynamics, masked_completion_loss=masked_loss)

    return InstructBaselineStats(
        dataset=dataset_stats,
        weights=_compute_weight_stats(model),
        training=training,
    )


def _compute_dpo_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
) -> DpoBaselineStats:
    texts = _extract_dpo_texts(records[:max_samples])
    prompts, chosens, rejecteds = zip(*texts) if texts else ([], [], [])

    prompt_lengths = _token_lengths(list(prompts), tokenizer)
    chosen_lengths = _token_lengths(list(chosens), tokenizer)
    rejected_lengths = _token_lengths(list(rejecteds), tokenizer)

    ratios = [c / r if r > 0 else 1.0 for c, r in zip(chosen_lengths, rejected_lengths)]
    all_texts = [p + " " + c for p, c in zip(prompts, chosens)]

    dataset_stats = DpoDatasetStats(
        total_tokens=sum(prompt_lengths) + sum(chosen_lengths) + sum(rejected_lengths),
        seq_length_distribution=_make_seq_dist([p + c for p, c in zip(prompt_lengths, chosen_lengths)]),
        near_duplicate_rate=_compute_near_duplicate_rate(list(prompts)),
        bits_per_byte=_compute_bits_per_byte(list(prompts), device),
        vocab_size=len(tokenizer),
        prompt_tokens=sum(prompt_lengths),
        chosen_tokens=sum(chosen_lengths),
        rejected_tokens=sum(rejected_lengths),
        chosen_length_distribution=_make_seq_dist(chosen_lengths),
        rejected_length_distribution=_make_seq_dist(rejected_lengths),
        chosen_rejected_length_ratio=float(np.mean(ratios)),
    )

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, all_texts, device)

    chosen_log_probs = _compute_log_probs(model, tokenizer, list(prompts), list(chosens), device)
    rejected_log_probs = _compute_log_probs(model, tokenizer, list(prompts), list(rejecteds), device)

    mean_chosen = float(np.mean(chosen_log_probs)) if chosen_log_probs else 0.0
    mean_rejected = float(np.mean(rejected_log_probs)) if rejected_log_probs else 0.0

    training = DpoTrainingDynamics(
        **base_dynamics,
        ref_log_prob_chosen=mean_chosen,
        ref_log_prob_rejected=mean_rejected,
        implicit_reward_gap=mean_chosen - mean_rejected,
    )

    return DpoBaselineStats(
        dataset=dataset_stats,
        weights=_compute_weight_stats(model),
        training=training,
    )


def _compute_grpo_stats(
    model, tokenizer, records: list[dict], device: str, max_samples: int,
    reward_functions=None,
) -> GrpoBaselineStats:
    prompts = _extract_grpo_texts(records[:max_samples])
    prompt_lengths = _token_lengths(prompts, tokenizer)

    dataset_stats = GrpoDatasetStats(
        total_tokens=sum(prompt_lengths),
        seq_length_distribution=_make_seq_dist(prompt_lengths),
        near_duplicate_rate=_compute_near_duplicate_rate(prompts),
        bits_per_byte=_compute_bits_per_byte(prompts, device),
        vocab_size=len(tokenizer),
        prompt_tokens=sum(prompt_lengths),
        prompt_length_distribution=_make_seq_dist(prompt_lengths),
    )

    base_dynamics = _compute_base_training_dynamics(model, tokenizer, prompts, device)

    # Compute baseline reward scores
    reward_scores: dict[str, float] = {}
    if reward_functions:
        completions = _generate_completions(model, tokenizer, prompts[:10], device)
        for rf in reward_functions:
            func_code = rf.reward_func if hasattr(rf, "reward_func") else str(rf)
            try:
                namespace = {}
                exec(func_code, namespace)
                func_name = [k for k in namespace if callable(namespace[k]) and k != "__builtins__"][0]
                scores = namespace[func_name](completions)
                reward_scores[func_name] = float(np.mean(scores))
            except Exception:
                reward_scores[func_code[:30]] = 0.0

    training = GrpoTrainingDynamics(**base_dynamics, baseline_reward_scores=reward_scores)

    return GrpoBaselineStats(
        dataset=dataset_stats,
        weights=_compute_weight_stats(model),
        training=training,
    )


def _generate_completions(model, tokenizer, prompts: list[str], device, max_new_tokens: int = 50) -> list[str]:
    """Generate short completions from the base model."""
    model.eval()
    completions = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            output_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
        generated = tokenizer.decode(output_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(generated)
    return completions


# --- Main entry point for text tasks ---

def compute_text_stats(
    model,
    tokenizer,
    data_records: list[dict],
    task_type: str = "instruct",
    max_samples: int = 100,
    reward_functions=None,
) -> BaselineStats:
    """Compute stats for text-based tasks (instruct, DPO, GRPO, chat)."""
    device = str(_get_model_device(model))

    if task_type == "chat":
        print("Computing chat stats...", flush=True)
        return _compute_instruct_stats(model, tokenizer, data_records, device, max_samples, text_extractor=_extract_chat_texts)
    elif task_type == "dpo":
        print("Computing DPO stats...", flush=True)
        return _compute_dpo_stats(model, tokenizer, data_records, device, max_samples)
    elif task_type == "grpo":
        print("Computing GRPO stats...", flush=True)
        return _compute_grpo_stats(model, tokenizer, data_records, device, max_samples, reward_functions)
    else:
        print(f"Computing instruct stats (task_type={task_type})...", flush=True)
        return _compute_instruct_stats(model, tokenizer, data_records, device, max_samples)
