"""Utilities for locating API key configuration files."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import config

KEY_CONFIG_FILENAMES: Sequence[str] = ("key.cfg", "keys.cfg")


def load_key_config() -> Optional[config.Config]:
    """Return the first configuration file that exists or ``None``."""
    for filename in KEY_CONFIG_FILENAMES:
        path = os.path.join(os.getcwd(), filename)
        if os.path.isfile(path):
            return config.Config(path)
    return None
