"""
Configuration loader for the tts-podcast tool.

Reads a YAML file and resolves keys ending with ``_env`` by looking up
the named environment variable.  Raises :exc:`ConfigError` when the file
is missing or a required environment variable is not set.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when configuration loading or env-var resolution fails."""


def _resolve_env_vars(data: Any) -> Any:
    """
    Recursively walk a parsed YAML structure and replace ``*_env`` keys.

    For every mapping key that ends with ``_env``, the value is treated
    as an environment variable name.  The ``*_env`` key is removed and a
    new key without the ``_env`` suffix is inserted with the env-var value.

    Parameters
    ----------
    data : Any
        A Python object produced by ``yaml.safe_load`` (dict, list, or scalar).

    Returns
    -------
    Any
        The same structure with all ``*_env`` keys resolved.

    Raises
    ------
    ConfigError
        If an environment variable referenced by a ``*_env`` key is not set.
    """
    if isinstance(data, dict):
        resolved: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and key.endswith("_env"):
                # The value is the name of an environment variable.
                env_var_name = str(value)
                env_value = os.environ.get(env_var_name)
                if env_value is None:
                    raise ConfigError(
                        f"Required environment variable '{env_var_name}' is not set "
                        f"(referenced by config key '{key}')."
                    )
                bare_key = key[: -len("_env")]
                resolved[bare_key] = env_value
            else:
                resolved[key] = _resolve_env_vars(value)
        return resolved
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load and return the configuration from a YAML file.

    Environment variable references (keys ending in ``_env``) are resolved
    automatically.

    Parameters
    ----------
    path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The fully resolved configuration mapping.

    Raises
    ------
    ConfigError
        If the file does not exist, cannot be parsed, or a referenced
        environment variable is not set.

    Examples
    --------
    >>> cfg = load_config("config.yaml")
    >>> cfg["gemini"]["text_model"]
    'gemini-2.0-flash'
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: '{config_path}'.")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse configuration file: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration file must contain a YAML mapping, got {type(raw).__name__}."
        )

    return _resolve_env_vars(raw)
