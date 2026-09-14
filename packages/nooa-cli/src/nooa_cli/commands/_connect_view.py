# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, responsive terminal presentation helpers for the Connect wizard."""

import os
import shutil
import sys
import textwrap

import click


def line(text, *, fg=None, bold=False, dim=False):
    width = max(24, min(84, shutil.get_terminal_size((80, 24)).columns - 4))
    for part in textwrap.wrap(text, width=width) or [""]:
        if "NO_COLOR" not in os.environ:
            part = click.style(part, fg=fg, bold=bold, dim=dim)
        click.echo("  " + part)


def intro(*, checks, output_tokens, budget_tokens):
    click.echo()
    line("NOOA  /  CONNECT", fg="bright_cyan", bold=True)
    line("Add a model to your workspace.", dim=True)
    click.echo()
    if checks:
        line("API checks may incur charges.", fg="yellow")
        line(f"{output_tokens:,} output tokens / call", dim=True)
        line(
            f"{budget_tokens:,} shared token budget"
            if budget_tokens is not None
            else "Up to 3 interface calls, then tools and each proposed reasoning level.",
            dim=True,
        )
        line("No retries. Caps are estimates, not billing limits.", dim=True)
    else:
        line("No generation calls. Listing and metadata may still be fetched.", dim=True)
    line("Skip paid checks: --no-probe    Help: F1", dim=True)
    if not sys.stdin.isatty():
        line("Manual setup: docs/model-configuration.md", dim=True)
        line("Agent skill: nooa-agent-authoring", dim=True)


def step(number, title):
    click.echo()
    line(f"{number} / 4  ·  {title}", fg="bright_cyan", bold=True)
    click.echo()


def model_details(model, *, output_tokens, edited=False):
    """Show published model information without changing any request settings."""

    def tokens(value):
        return (
            f"{value:,} tokens"
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
            else "Not listed"
        )

    reasoning = model.get("reasoning") or {}
    levels = reasoning.get("supported_efforts") or []
    click.echo()
    line(f"Model details · {model['id']}", fg="bright_cyan", bold=True)
    for label, value in (
        ("Context window", tokens(model.get("context_length"))),
        (
            "Maximum reply length",
            tokens((model.get("top_provider") or {}).get("max_completion_tokens")),
        ),
        ("Reasoning levels", ", ".join(levels) if levels else "Not listed"),
        ("Default reasoning", reasoning.get("default_effort") or "Not listed"),
        ("Setup check limit", f"{output_tokens:,} tokens per reply (checks only)"),
    ):
        line(f"{label:<25} {value}")
    line(
        "Your edited settings."
        if edited
        else "Source: OpenRouter model listing. Your server may use different limits.",
        dim=True,
    )
    line("Setup checks do not measure maximum limits.", dim=True)
    click.echo()
