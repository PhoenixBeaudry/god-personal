import json
from pathlib import Path


def scrub_model_identity(model_dir: str) -> None:
    """Remove model identity fields from config files in a downloaded model directory."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        if "_name_or_path" in config:
            del config["_name_or_path"]
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Scrubbed _name_or_path from {config_path}")
    except Exception as e:
        print(f"Warning: Failed to scrub model identity from {config_path}: {e}")
