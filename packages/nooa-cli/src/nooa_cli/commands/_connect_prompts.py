# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Editable terminal prompts; the onboarding library has no terminal dependencies.

Only suggested names enter completion, never environment values. Each prompt
has disposable history, and secrets have neither history nor completion. Pipes
use Click's line input so scripted answers continue to work without a terminal.
"""

import os
import sys

import click


def environment_names(defaults=()):
    """Suggest variable names, including conventional names not yet exported."""
    return sorted(set(os.environ).union(defaults))


def prompt(text, *, default=None, choices=(), suggestions=(), hide_input=False, show_default=True):
    """Read one editable answer with a scrolling, single-column completion menu."""
    choices = tuple(choices)
    if not sys.stdin.isatty():
        while True:
            value = click.prompt(
                text, default=default, hide_input=hide_input, show_default=show_default
            )
            if not choices or value in choices:
                return value
            # Click.Choice includes every model ID in its prompt and error.
            click.echo(
                "Choose an available value; check the spelling of the model or provider name.",
                err=True,
            )

    from prompt_toolkit import prompt as terminal_prompt
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import DummyHistory
    from prompt_toolkit.validation import Validator

    words = list(dict.fromkeys(choices or suggestions))
    completer = (
        WordCompleter(words, ignore_case=True, match_middle=True, sentence=True)
        if words and not hide_input
        else None
    )
    validator = Validator.from_callable(
        lambda value: value in choices if choices else bool(value) or default == "",
        error_message="Choose a matching value." if choices else "Enter a value.",
        move_cursor_to_end=False,
    )
    try:
        return terminal_prompt(
            text + ": ",
            default="" if default is None else str(default),
            completer=completer,
            complete_while_typing=True,
            reserve_space_for_menu=8,
            history=DummyHistory(),
            is_password=hide_input,
            validator=validator,
            validate_while_typing=False,
        )
    except (EOFError, KeyboardInterrupt):
        raise click.Abort() from None


def confirm(text, *, default):
    """Keep approval explicit; completion never submits an answer."""
    if not sys.stdin.isatty():
        return click.confirm(text, default=default)
    value = prompt(
        text + (" [Y/n]" if default else " [y/N]"),
        default="",
        choices=("yes", "no", "y", "n", ""),
    )
    return default if not value else value in {"yes", "y"}
