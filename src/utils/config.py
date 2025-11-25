import yaml
from types import SimpleNamespace


def _to_namespace(obj):
    """Recursively convert dicts to SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_to_namespace(i) for i in obj]
    else:
        return obj


def load_config(path: str):
    """Load YAML config and return as dot-accessible Namespace."""
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return _to_namespace(cfg_dict)
