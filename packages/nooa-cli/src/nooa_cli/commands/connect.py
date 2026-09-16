# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy CLI entry point; Connect is owned by UnifiedLLM."""

import click


@click.command(
    add_help_option=False,
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def command(ctx):
    """Walk through model setup, check the connection, and save an alias."""
    from nooa.unifiedllm.connect.cli import command as connect_command

    result = connect_command.main(args=ctx.args, prog_name=ctx.command_path, standalone_mode=False)
    # Click returns explicit Exit codes in non-standalone mode. Preserve them
    # for scripted stage calls instead of treating a nonzero return as success.
    if isinstance(result, int):
        ctx.exit(result)
    return result
