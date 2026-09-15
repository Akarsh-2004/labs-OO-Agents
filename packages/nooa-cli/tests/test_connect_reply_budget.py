# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human presets and scripted saves use the same actual reply limit."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command


@pytest.mark.parametrize(
    "choice,expected", [("short", 2048), ("coding", 8192), ("long", 32768), ("custom\n512", 512)]
)
def test_wizard_reply_budget_is_a_runtime_cap(tmp_path, monkeypatch, choice, expected):
    from nooa import llm_config

    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [])
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--as",
            "local",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "responses",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--no-probe",
            "--output",
            str(path),
        ],
        input=choice + "\ny\n",
    )
    assert result.exit_code == 0, result.output
    entry = yaml.safe_load(path.read_text())["models"]["local"]
    assert entry["max_tokens"] == expected
    assert entry["include"] == ["reasoning.encrypted_content"]
    assert entry["store"] is False


def test_stage_save_fills_defaults_and_reports_shadow(tmp_path, monkeypatch):
    from nooa import llm_config

    source = tmp_path / "override.yaml"
    source.write_text("models: {local: {model_name: openai/old}}\n")
    monkeypatch.setenv("NEMO_OO_LLM_CONFIG", str(source))
    monkeypatch.setattr(llm_config, "llm_config_chain", lambda: [source])
    document = tmp_path / "entry.json"
    document.write_text(
        json.dumps(
            {"alias": "local", "entry": {"model_name": "openai/model", "client_type": "responses"}}
        )
    )
    destination = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(document), "--output", str(destination)]
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["data"]["shadowed_by"] == str(source)
    entry = report["data"]["entry"]
    assert entry["max_tokens"] == 8192
    assert entry["include"] == ["reasoning.encrypted_content"]
    assert yaml.safe_load(destination.read_text())["models"]["local"] == entry


def test_scripted_plan_honours_explicit_reply_cap():
    result = CliRunner().invoke(
        command,
        [
            "model",
            "--stage",
            "plan",
            "--endpoint",
            "https://api.test/v1",
            "--api-style",
            "responses",
            "--max-tokens",
            "1234",
        ],
    )
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)["data"]
    assert plan["entry"]["max_tokens"] == 1234
    assert plan["probes"][0]["body"]["max_output_tokens"] == 200
