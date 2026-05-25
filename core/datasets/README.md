# Datasets

Shared dataset assets and adapters live here.

- `adapters.py` rewrites text, DPO, GRPO, and environment datasets into the column names expected by training.
- `diffusion.py` prepares image zip uploads into the folder shape expected by diffusion trainers.
- `whitelist.py` validates miner-requested Hugging Face datasets against `whitelisted_sft_datasets.json`.
- `examples/` contains small local sample datasets.
- `images/` is the default local image-prep workspace used by diffusion training helpers.

