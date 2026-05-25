import hashlib
import os


def get_anonymous_model_dir(model_id: str) -> str:
    """Convert a HF model ID to an anonymous salted hash for cache directory naming.

    The salt comes from MODEL_HASH_SALT env var (set on validators, never exposed to miners).
    Without the salt, miners cannot reverse the hash to discover the model identity.
    """
    salt = os.environ.get("MODEL_HASH_SALT", "")
    return hashlib.sha256((salt + model_id).encode()).hexdigest()[:16]
