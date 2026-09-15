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


def shadowing_source(alias, path):
    """Name a currently effective file that would override this destination."""
    import os
    from pathlib import Path

    from nooa.llm_config import bundled_config_paths
    from nooa.paths import get_project_dir, get_user_dir

    found = entries().get(alias)
    if not found or found[1].resolve() == path.resolve():
        return None
    # Include missing conventional files so a newly created user/project file
    # receives its real priority. Unknown --output paths are not auto-loaded.
    paths = [
        *bundled_config_paths(),
        get_user_dir("llm_config.yaml"),
        get_project_dir("llm_config.yaml"),
        *(
            Path(p.strip()).expanduser()
            for p in os.environ.get("NEMO_OO_LLM_CONFIG", "").split(",")
            if p.strip()
        ),
    ]
    priority = {p.resolve(): i for i, p in enumerate(paths)}
    target = priority.get(path.resolve())
    source = priority.get(found[1].resolve())
    if target is None or source is None or source > target:
        return str(found[1])
    return None
