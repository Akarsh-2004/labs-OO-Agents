# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic handoffs explain reproduction and sources without exposing credentials."""

import shlex

from nooa_cli.commands._connect_registry import diagnostic_context

from nooa import connect


def test_target_and_effective_source_are_distinct(tmp_path, monkeypatch):
    from nooa import llm_config

    target = tmp_path / "target file.yaml"
    override = tmp_path / "override.yaml"
    override.write_text("models: {saved: {model_name: openai/model}}\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [override])
    monkeypatch.setenv("CHECK_KEY", "private-value")
    context = diagnostic_context(
        target=target,
        alias="saved",
        model="model; not a command",
        endpoint="https://api.test/v1",
        api_key_env="CHECK_KEY",
        budget=131072,
        remaining=128936,
        stage="interfaces",
    )
    assert context["target_file"] == str(target)
    assert context["effective_alias_source"] == str(override)
    assert context["registry_files"] == [str(override)]
    assert context["credential_available"] is True
    assert shlex.split(context["rerun_command"])[4] == "model; not a command"
    assert "private-value" not in repr(context)
    prompt = connect.diagnostic_prompt(
        "interfaces", {}, {}, run_context={**context, "api_key": "secret"}
    )
    assert '"api_key"' not in prompt
    assert "git clone --depth 1" in prompt
    assert "does not authorize additional paid calls" in prompt


def test_context_does_not_mask_broken_yaml_or_leak_pasted_key(tmp_path, monkeypatch):
    from nooa import llm_config

    broken = tmp_path / "broken.yaml"
    broken.write_text("models: [invalid: : private-secret\n")
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [broken])
    context = diagnostic_context(
        target=tmp_path / "private-secret.yaml",
        alias="saved",
        api_key="private-secret",
        model="model",
        endpoint="https://u:private-secret@api.test/v1",
        remaining=0,
        stage="interfaces",
    )
    assert "private-secret" not in repr(context)
    assert "rerun_command" not in context
    assert context["credential_source"] == "pasted"
