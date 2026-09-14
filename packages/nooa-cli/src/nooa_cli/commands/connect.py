# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal approval and display for the shared nooa.connect library."""

import click


@click.command()
@click.argument("model", required=False)
@click.option("--as", "alias", required=True, help="Local model alias to save.")
@click.option("--endpoint", required=True, help="API base URL, including /v1 if required.")
@click.option("--api-style", type=click.Choice(["chat", "responses", "anthropic"]), required=True)
@click.option(
    "--api-key-env",
    default="OPENAI_API_KEY",
    show_default=True,
    help="Environment variable name, never the key itself.",
)
@click.option("--catalogue-model", help="Explicit OpenRouter model ID to use as metadata.")
@click.option("--prompt-key", is_flag=True, help="Read a masked, temporary key; never save it.")
@click.option("--no-catalogue", is_flag=True, help="Do not fetch public model metadata.")
@click.option(
    "--reasoning-template",
    type=click.Choice(["effort", "adaptive", "budget", "toggle", "thinking"]),
)
@click.option("--levels", help="Comma-separated candidate labels to probe.")
@click.option(
    "--levels-file",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML mapping from labels to complete request settings.",
)
@click.option(
    "--context-window",
    type=click.IntRange(min=1),
    help="User-supplied context limit, not a request allocation.",
)
@click.option(
    "--probe", type=click.Choice(["all", "minimal", "none"]), default="all", show_default=True
)
@click.option("--no-probe", is_flag=True, help="Save an untested entry without model calls.")
@click.option("--budget-tokens", type=click.IntRange(min=1), default=4096, show_default=True)
@click.option("--output-tokens", type=click.IntRange(1, 4096), default=200, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Registry path; defaults to the user llm_config.yaml.",
)
@click.option("--yes", is_flag=True, help="Approve the displayed plan and write without prompting.")
def command(
    model,
    alias,
    endpoint,
    api_style,
    api_key_env,
    prompt_key,
    catalogue_model,
    no_catalogue,
    reasoning_template,
    levels,
    levels_file,
    context_window,
    probe,
    no_probe,
    budget_tokens,
    output_tokens,
    output,
    yes,
):
    """Configure MODEL using a reviewed plan and optional bounded probes.

    MODEL is the exact endpoint model ID, without the extra LiteLLM routing
    prefix. Catalogue data suggests settings; only the selected endpoint is
    probed. The same library supports interactive TUI onboarding.
    """
    import asyncio
    import os
    from pathlib import Path

    import httpx
    import yaml

    from nooa import connect
    from nooa.paths import get_user_dir

    path = Path(output) if output else get_user_dir("llm_config.yaml")
    try:
        if not model and yes:
            raise click.UsageError("Supply MODEL with --yes; endpoint discovery needs a selection.")
        api_key = (
            click.prompt("API key (used only for this setup)", hide_input=True)
            if prompt_key
            else os.environ.get(api_key_env)
            if api_key_env
            else None
        )
        if not model:
            found = asyncio.run(connect.discover(endpoint, api_style=api_style, api_key=api_key))
            endpoint = found.api_base
            names = [item["id"] for item in found.models]
            click.echo("Endpoint models:\n" + "\n".join(names))
            model = click.prompt("Model", type=click.Choice(names))
        existing = None
        if path.exists():
            text = path.read_text()
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
                raise click.ClickException("Registry must contain a models mapping.")
            existing = data.get("models", {}).get(alias)
            if alias in data.get("models", {}):
                click.echo(f"Warning: saving will overwrite model {alias!r} in {path}.", err=True)
                if not yes and not click.confirm("Replace this model?", default=False):
                    return
        candidate = None
        if no_catalogue and catalogue_model:
            raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
        if not no_catalogue:
            models = asyncio.run(connect.catalogue())
            matches = (
                [item for item in models if item.get("id") == catalogue_model]
                if catalogue_model
                else connect.match_models(model, models)
            )
            if catalogue_model and not matches:
                raise click.ClickException("Requested catalogue model was not found.")
            if len(matches) == 1:
                click.echo(
                    f"Catalogue candidate: {matches[0]['id']} (not proof of the endpoint's capabilities)"
                )
                if yes or click.confirm("Use this candidate's metadata?", default=True):
                    candidate = matches[0]
            elif matches:
                click.echo(
                    "Possible catalogue models: " + ", ".join(item["id"] for item in matches)
                )
                if yes:
                    raise click.ClickException(
                        "Ambiguous match: choose --catalogue-model or --no-catalogue."
                    )
                selected = click.prompt(
                    "Catalogue model (blank leaves it unknown)", default="", show_default=False
                )
                if selected:
                    candidate = next((item for item in matches if item["id"] == selected), None)
                    if candidate is None:
                        raise click.ClickException("Choose one of the displayed model IDs.")
            else:
                click.echo("No catalogue match; model limits and reasoning levels remain unknown.")
        if levels_file and (levels or reasoning_template):
            raise click.UsageError(
                "Use either --levels-file or --reasoning-template with --levels."
            )
        patches = yaml.safe_load(Path(levels_file).read_text()) if levels_file else None
        if levels or reasoning_template:
            if not reasoning_template or not levels:
                raise click.UsageError(
                    "--reasoning-template and --levels must be supplied together."
                )
            patches = {
                label.strip(): connect.reasoning_settings(
                    reasoning_template, api_style, label.strip()
                )
                for label in levels.split(",")
            }
        proposal = connect.plan(
            alias,
            model,
            api_style,
            endpoint,
            api_key_env,
            catalogue=candidate,
            reasoning_levels=patches,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            existing_entry=existing,
        )
        if context_window:
            proposal.entry["context_window"] = context_window
            proposal.entry["provenance"]["context_window"] = {
                "source": "user",
                "value": context_window,
            }
        if patches:
            proposal.entry["provenance"]["reasoning_levels"] = {
                "source": "user",
                "template": reasoning_template,
            }
        if "context_window" not in proposal.entry:
            click.echo(
                "Context window unknown: the runtime's existing fallback applies; set --context-window if known."
            )
        approval = "none" if no_probe else probe
        click.echo(yaml.safe_dump({"models": {alias: proposal.entry}}, sort_keys=False))
        price = (
            "unknown"
            if proposal.price_estimate is None
            else f"~${proposal.price_estimate:.6f} at catalogue prices"
        )
        click.echo(
            f"Plan: {len(proposal.probes)} candidate calls, {output_tokens} output tokens per call; approval: {approval}."
        )
        click.echo(
            f"Estimated total: {proposal.token_estimate} tokens; budget: {budget_tokens}; price: {price}."
        )
        click.echo(
            "No retries or capacity probes. Estimates are not billing limits: endpoints can ignore output caps."
        )
        if (
            approval != "none"
            and not yes
            and not click.confirm("Run these paid probes?", default=False)
        ):
            approval = "none"
        result = asyncio.run(connect.run(proposal, approved=approval, api_key=api_key))
        for name, outcome in result.entry["provenance"]["probes"].items():
            click.echo(
                f"{name}: {outcome['outcome']}"
                + (f" ({outcome['reason']})" if "reason" in outcome else "")
            )
        if yes or click.confirm(f"Write model entry to {path}?", default=True):
            connect.write(result.entry, path, alias=alias)
            click.echo(f"Saved {alias} to {path}.")
            if output:
                click.echo(
                    "For a custom path, include it in NEMO_OO_LLM_CONFIG or reload_registry(path)."
                )
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError) as exc:
        raise click.ClickException(str(exc)) from None
