# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable terminal layout without changing the setup's cost or save policy."""

import os

import click
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view


def test_intro_is_wrapped_and_keeps_cost_notice_and_manual_routes(monkeypatch):
    monkeypatch.setattr(view.shutil, "get_terminal_size", lambda *args: os.terminal_size((58, 24)))

    @click.command()
    def command():
        view.intro(checks=True, output_tokens=200, budget_tokens=4096)

    result = CliRunner().invoke(command)
    assert result.exit_code == 0, result.output
    assert all(len(line) <= 58 for line in result.output.splitlines())
    assert "may incur charges" in result.output
    assert "200" in result.output and "4,096" in result.output
    assert "--no-probe" in result.output
    assert "F1" in result.output
    assert "docs/model-configuration.md" in result.output
    assert "nooa-agent-authoring" in result.output


def test_no_color_keeps_readable_titles_and_warning(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")

    @click.command()
    def command():
        view.intro(checks=True, output_tokens=200, budget_tokens=4096)
        view.step(1, "Connection")

    result = CliRunner().invoke(command, color=True)
    assert "\x1b[" not in result.output
    assert "Connection" in result.output
    assert "may incur charges" in result.output
