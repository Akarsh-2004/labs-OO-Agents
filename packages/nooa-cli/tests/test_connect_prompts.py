# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real terminal editor, including completion and cursor keys."""

import pytest
from nooa_cli.commands import _connect_prompts as prompts
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


@pytest.fixture(autouse=True)
def terminal_type(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def answer(monkeypatch, keys, **kwargs):
    import prompt_toolkit
    from prompt_toolkit.application import get_app

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        if "\t" in keys:
            # Wait for the menu, as a user would, before selecting a completion.
            before, after = keys.split("\t", 1)
            original = prompt_toolkit.prompt

            def run(*args, **options):
                def ready():
                    buffer = get_app().current_buffer

                    def menu_opened(_):
                        if buffer.complete_state:
                            buffer.on_completions_changed -= menu_opened
                            pipe.send_text("\t" + after)

                    buffer.on_completions_changed += menu_opened
                    pipe.send_text(before)

                return original(*args, pre_run=ready, **options)

            monkeypatch.setattr(prompt_toolkit, "prompt", run)
        else:
            pipe.send_text(keys)
        return prompts.prompt("Choose", **kwargs)


def test_model_completion_matches_inside_long_ids(monkeypatch):
    assert (
        answer(monkeypatch, "qwen\t\r", choices=["vendor/model-one", "vendor/qwen-model"])
        == "vendor/qwen-model"
    )


def test_environment_completion_only_uses_names(monkeypatch):
    monkeypatch.setenv("CONNECT_COMPLETION_KEY", "secret-never-completed")
    names = prompts.environment_names(["ANOTHER_KEY"])
    assert "CONNECT_COMPLETION_KEY" in names
    assert "ANOTHER_KEY" in names
    assert "secret-never-completed" not in names
    assert (
        answer(monkeypatch, "CONNECT_COMPLETION\t\r", suggestions=names) == "CONNECT_COMPLETION_KEY"
    )


@pytest.mark.parametrize(
    "keys,expected",
    [
        ("\x1b[D\x1b[DZ\r", "abZcd"),  # Left moves within the prefilled value.
        ("\x1b[H\x1b[C\x1b[3~\x1b[F!\r", "acd!"),  # Home, Right, Delete, End.
    ],
)
def test_prefilled_text_is_editable(monkeypatch, keys, expected):
    assert answer(monkeypatch, keys, default="abcd") == expected


def test_completion_does_not_submit_a_confirmation(monkeypatch):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\r")
        assert prompts.confirm("Spend?", default=False) is False


def test_confirmation_accepts_yes_without_deleting_default(monkeypatch):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("y\r")
        assert prompts.confirm("Spend?", default=False) is True


def test_secret_prompt_has_no_completer_or_history(monkeypatch):
    import prompt_toolkit

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return "secret"

    monkeypatch.setattr(prompt_toolkit, "prompt", capture)
    assert prompts.prompt("Key", hide_input=True, suggestions=["must-not-appear"]) == "secret"
    assert seen["is_password"] is True
    assert seen["completer"] is None
    assert list(seen["history"].get_strings()) == []


def test_cancel_uses_click_abort(monkeypatch):
    import click

    with pytest.raises(click.Abort):
        answer(monkeypatch, "\x03")
