# Validator Tasks

This package owns task preparation before work is sent to the trainer.

- `task_prep.py` prepares text and image datasets.
- `details.py` shapes API task details and hides sensitive in-flight data.
- `augmentation.py` decides when optional model augmentation is enabled.
- `dataset_columns.py` validates user-supplied dataset column mappings.
- `dataset_mapping.py` standardizes multi-dataset text samples to validator column contracts.
- `model_prep.py` dispatches augmentation and baseline statistics work to trainers.
- `requests.py` converts prepared validator tasks into trainer request payloads.
- `reward_functions.py` and `affine_reward_functions.py` validate and package GRPO reward functions.
- `yarn_extension.py` prepares YaRN-extended models before trainer requests are built.
- `synthetic_scheduler.py` and `diffusion_synth.py` create synthetic tasks.
- `image_synth/` contains image prompt and dataset generation support.
- `example_prompts.json` stores seed prompts for image task generation.
