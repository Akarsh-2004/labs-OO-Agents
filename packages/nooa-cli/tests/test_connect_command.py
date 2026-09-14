# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
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
    monkeypatch.setattr(connect, "run_steps", forbidden)
    path = tmp_path / "models.yaml"
    original = "models: {local: {model_name: openai/old}}\n"
    path.write_text(original)
    result = CliRunner().invoke(command, args(path), input="n\n")
    assert result.exit_code == 0, result.output
    assert path.read_text() == original


def test_prompted_key_is_passed_to_probe_but_never_saved(tmp_path, monkeypatch):
    from nooa import connect

    calls = []

    async def run(proposal, *, approved, api_key=None):
        calls.append(approved)
        assert api_key == "temporary-secret"
        yield connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "run_steps", run)
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command, [*args(path), "--prompt-key", "--yes"], input="temporary-secret\n"
    )
    assert result.exit_code == 0, result.output
    assert calls == ["none"]
    assert "temporary-secret" not in result.output + path.read_text()


def test_bare_command_walks_through_setup_and_checks_inline(tmp_path, monkeypatch):
    import click
    import httpx

    from nooa import connect, paths

    output, requests = [], []
    real_echo = click.echo
    real_client = httpx.AsyncClient
    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    monkeypatch.delenv("CONNECT_WIZARD_KEY", raising=False)

    def echo(message=None, **kwargs):
        output.append(str(message))
        return real_echo(message, **kwargs)

    def handle(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer temporary-secret"
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "example-model"}]})
        # Feedback appears before the HTTP operation, not only at the end.
        assert any("Checking" in line for line in output)
        return httpx.Response(200, json={"choices": [{"message": {"content": "323"}}]})

    async def catalogue():
        return []

    monkeypatch.setattr(click, "echo", echo)
    monkeypatch.setattr(connect, "catalogue", catalogue)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)
    )
    result = CliRunner().invoke(
        command,
        [],
        input=(
            "custom\nhttps://api.test/v1\nCONNECT_WIZARD_KEY\ntemporary-secret\n"
            "example-model\nchat\nmy-model\ny\ny\n"
        ),
    )
    assert result.exit_code == 0, result.output
    assert [r.method for r in requests] == ["GET", "POST", "POST"]
    entry = yaml.safe_load((tmp_path / "llm_config.yaml").read_text())["models"]["my-model"]
    assert entry["model_name"] == "openai/example-model"
    assert "temporary-secret" not in result.output + yaml.safe_dump(entry)
    assert result.output.index("Checking routing") < result.output.index("routing: accepted")
    assert result.output.index("routing: accepted") < result.output.index("Checking tools")


def test_script_mode_requires_missing_options_without_prompting():
    result = CliRunner().invoke(command, ["--yes"])
    assert result.exit_code == 2
    assert "--endpoint" in result.output


def test_bare_command_cancel_before_endpoint_does_nothing(tmp_path, monkeypatch):
    from nooa import paths

    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    result = CliRunner().invoke(command, [], input="")
    assert result.exit_code == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "provider,base,style,key_env",
    [
        ("nvidia", "https://integrate.api.nvidia.com/v1", "chat", "NVIDIA_API_KEY"),
        ("openai", "https://api.openai.com/v1", "responses", "OPENAI_API_KEY"),
        ("anthropic", "https://api.anthropic.com/v1", "anthropic", "ANTHROPIC_API_KEY"),
        (
            "google",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "chat",
            "GEMINI_API_KEY",
        ),
        ("openrouter", "https://openrouter.ai/api/v1", "chat", "OPENROUTER_API_KEY"),
    ],
)
def test_provider_menu_fills_connection_defaults(
    tmp_path, monkeypatch, provider, base, style, key_env
):
    from nooa import connect, paths

    seen = []

    async def discover(endpoint, **kwargs):
        seen.append((endpoint, kwargs))
        return connect.Discovery(endpoint, ({"id": "test-model"},))

    monkeypatch.setattr(paths, "get_user_dir", lambda name: tmp_path / name)
    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.setenv(key_env, "preset-test-key")
    result = CliRunner().invoke(
        command, ["--no-catalogue", "--no-probe"], input=f"{provider}\ntest-model\n\nmy-model\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert seen == [(base, {"api_style": style, "api_key": "preset-test-key"})]
    entry = yaml.safe_load((tmp_path / "llm_config.yaml").read_text())["models"]["my-model"]
    assert (entry["api_base"], entry["api_style"], entry["api_key_env"]) == (base, style, key_env)
    assert "preset-test-key" not in result.output
    assert "Model server URL:" not in result.output
    assert "Key environment variable" not in result.output
    assert result.output.index("Model:") < result.output.index("API format [")


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
def test_mixed_endpoint_selects_model_before_request_interface(tmp_path, monkeypatch, style):
    import httpx

    real_client = httpx.AsyncClient
    seen = []

    def handle(request):
        seen.append(request)
        assert request.method == "GET"
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200, json={"data": [{"id": "vendor-a/model"}, {"id": "vendor-b/model"}]}
        )

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handle), **kw)
    )
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "--endpoint",
            "https://api.test/v1",
            "--api-key-env",
            "",
            "--no-catalogue",
            "--no-probe",
            "--output",
            str(path),
        ],
        input=f"vendor-b/model\n{style}\nselected\ny\n",
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert result.output.index("Model:") < result.output.index("API format [")
    entry = yaml.safe_load(path.read_text())["models"]["selected"]
    assert entry["api_style"] == style
    assert entry["model_name"].endswith("/vendor-b/model")


def test_provider_flag_supports_scripted_setup(tmp_path):
    path = tmp_path / "models.yaml"
    result = CliRunner().invoke(
        command,
        [
            "test-model",
            "--provider",
            "nvidia",
            "--as",
            "local",
            "--no-catalogue",
            "--no-probe",
            "--yes",
            "--output",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text())["models"]["local"]["api_key_env"] == "NVIDIA_API_KEY"


def test_large_model_list_and_invalid_choice_do_not_flood_terminal(tmp_path, monkeypatch):
    from nooa import connect

    async def discover(endpoint, **kwargs):
        return connect.Discovery(endpoint, tuple({"id": f"vendor/model-{i}"} for i in range(1000)))

    monkeypatch.setattr(connect, "discover", discover)
    result = CliRunner().invoke(
        command, args(tmp_path / "models.yaml")[1:], input="wrong\nvendor/model-42\ny\n"
    )
    assert result.exit_code == 0, result.output
    assert "1000 model(s)" in result.output
    assert "vendor/model-999" not in result.output
    assert "vendor/model-998" not in result.output
    assert len(result.output) < 6000
