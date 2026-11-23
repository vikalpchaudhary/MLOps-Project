from pathlib import Path

import yaml


def load_params():
    """Read project settings when a training command starts."""
    with Path("params.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)
