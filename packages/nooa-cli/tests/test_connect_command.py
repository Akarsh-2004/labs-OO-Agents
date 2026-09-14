# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command


def args(path):
    return [
        "wire/model",
        "--as",
        "local",
        "--endpoint",
        "https://api.test/v1",
        "--api-style",
        "chat",
        "--api-key-env",
        "CONNECT_TEST_KEY",
        "--no-catalogue",
        "--no-probe",
        "--output",
        str(path),
    ]


def test_offline_cli_needs_no_key_and_writes_generated_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"
    assert "not_probed" in result.output


def test_declining_final_write_leaves_no_file(tmp_path):
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(command, args(path), input="n\n")
    assert result.exit_code == 0, result.output
    assert not path.exists()


def test_explicit_levels_are_written_as_request_blocks(tmp_path):
    path = tmp_path / "connected.yaml"
    result = CliRunner().invoke(
        command, [*args(path), "--yes", "--reasoning-template", "effort", "--levels", "low,high"]
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["reasoning_levels"] == {
        "low": {"reasoning_effort": "low"},
        "high": {"reasoning_effort": "high"},
    }


def test_existing_hand_written_alias_is_replaced_with_warning(tmp_path):
    path = tmp_path / "connected.yaml"
    runner = CliRunner()
    path.write_text("# My models\nmodels:\n  local: {model_name: openai/old}\n")
    result = runner.invoke(command, [*args(path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "Warning:" in result.output
    assert "local" in result.output
    assert str(path) in result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"
    assert path.read_text().startswith("# My models\n")


def test_endpoint_first_flow_uses_shared_discovery(tmp_path, monkeypatch):
    from nooa import connect

    calls = []

    async def discover(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return connect.Discovery("https://api.test/v1", ({"id": "wire/model"},))

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(command, args(path)[1:], input="wire/model\ny\n")
    assert result.exit_code == 0, result.output
    assert calls == [("https://api.test/v1", {"api_style": "chat", "api_key": None})]
    assert yaml.safe_load(path.read_text())["models"]["local"]["model_name"] == "openai/wire/model"


def test_masked_key_is_transient_and_cancel_does_not_write(tmp_path, monkeypatch):
    import os

    from nooa import connect

    calls = []

    async def discover(endpoint, **kwargs):
        calls.append(endpoint)
        assert kwargs["api_key"] == "temporary-secret"
        return connect.Discovery(endpoint, ({"id": "wire/model"},))

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.delenv("CONNECT_TEST_KEY", raising=False)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path)[1:], "--prompt-key"], input="temporary-secret\n"
    )
    assert result.exit_code != 0  # EOF cancels model selection.
    assert calls == ["https://api.test/v1"]
    assert "temporary-secret" not in result.output
    assert not path.exists()
    assert "CONNECT_TEST_KEY" not in os.environ


def test_declining_replace_stops_before_discovery_or_paid_work(tmp_path, monkeypatch):
    from nooa import connect

    async def forbidden(*args, **kwargs):
        raise AssertionError("Cancelled replacement must not call the endpoint")

    monkeypatch.setattr(connect, "catalogue", forbidden)
    monkeypatch.setattr(connect, "run", forbidden)
    path = tmp_path / "models.yaml"
    original = "models: {local: {model_name: openai/old}}\n"
    path.write_text(original)
    result = CliRunner().invoke(command, args(path), input="n\n")
    assert result.exit_code == 0, result.output
    assert path.read_text() == original


def test_prompted_key_is_passed_to_probe_but_never_saved(tmp_path, monkeypatch):
    from nooa import connect

    async def run(proposal, *, approved, api_key=None):
        assert api_key == "temporary-secret"
        return connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "run", run)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path), "--prompt-key", "--yes"], input="temporary-secret\n"
    )
    assert result.exit_code == 0, result.output
    assert "temporary-secret" not in result.output + path.read_text()
