# Adapted for this project; original logic reviewed and re-validated.
"""Stable hashing utility for experiment configuration caching.

Produces a deterministic hash string from a config dictionary,
suitable for keying cached kernel matrices and result files.
"""

import hashlib
import json
from typing import Any, Dict


def stable_hash(config: Dict[str, Any]) -> str:
    """Compute a stable SHA-256 hash of a configuration dictionary.

    The dictionary is serialized to JSON with sorted keys and no
    whitespace to ensure identical configs always produce the same hash,
    regardless of insertion order.

    Args:
        config: Dictionary of configuration parameters.

    Returns:
        A 16-character hex string (first 64 bits of SHA-256).
    """
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"),
                            default=_default_serializer)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _default_serializer(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)
