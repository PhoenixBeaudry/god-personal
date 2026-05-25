import hashlib
import os


def get_anonymous_model_dir(model_id: str) -> str:
    """Convert a model ID to a salted cache directory name without exposing the original ID."""
    salt = os.environ.get("MODEL_HASH_SALT", "")
    return hashlib.sha256((salt + model_id).encode()).hexdigest()[:16]
