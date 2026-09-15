# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the registry's actual layers for editing and credential suggestions."""


def entries(extra_path=None):
    """Return resolved entries and their defining files, without registering clients."""
    from pathlib import Path

    import yaml

    from nooa.llm_config import llm_config_chain

    paths = llm_config_chain()
    if extra_path is not None and extra_path.exists():
        paths = [p for p in paths if p.resolve() != extra_path.resolve()] + [extra_path]
    resolved = {}
    for path in paths:
        with Path(path).open() as source:
            data = yaml.safe_load(source) or {}
        if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
            raise ValueError(f"Registry {path} must contain a models mapping")
        for alias, entry in data.get("models", {}).items():
            if isinstance(alias, str) and isinstance(entry, dict):
                resolved[alias] = (entry, Path(path))
    return resolved


def credential_names(registry, endpoint):
    """Match a server, allowing its root and terminal /v1 forms; never return values."""
    from nooa.connect import normalize_endpoint

    def normalized(value):
        return normalize_endpoint(value).removesuffix("/v1")

    target = normalized(endpoint)
    names = []
    for entry, _ in registry.values():
        address, name = entry.get("api_base"), entry.get("api_key_env")
        if not isinstance(address, str) or not isinstance(name, str):
            continue
        try:
            matches = normalized(address) == target
        except ValueError:
            continue
        if matches and name not in names:
            names.append(name)
    return names
