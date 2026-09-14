# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal approval and display for the shared nooa.connect library."""

import click


@click.command()
@click.argument("model", required=False)
@click.option(
    "--provider",
    help="Connection preset: nvidia, openai, anthropic, google, openrouter, or custom.",
)
@click.option("--as", "alias", help="Local model alias to save (otherwise prompted).")
@click.option("--endpoint", help="API base URL (otherwise prompted).")
@click.option("--api-style", type=click.Choice(["chat", "responses", "anthropic"]))
@click.option(
    "--api-key-env",
    help="Environment variable name, never the key itself (otherwise prompted).",
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
@click.option(
    "--budget-tokens",
    type=click.IntRange(min=1),
    help="Shared token budget; by default reserve enough for all selected checks.",
)
@click.option("--output-tokens", type=click.IntRange(1, 4096), default=200, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Registry path; defaults to the user llm_config.yaml.",
)
@click.option("--yes", is_flag=True, help="Save without prompting; supply the connection options.")
def command(
    model,
    provider,
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
    """Walk through model setup, check the connection, and save an alias.

    Run `uv run nooa connect` with no arguments for guided setup. Flags prefill
    the answers; --yes requires MODEL, --as and either --provider or an
    explicit --endpoint and --api-style.
    MODEL is the exact endpoint model ID, without a LiteLLM routing prefix.
    """
    import asyncio
    import os
    from contextlib import aclosing
    from dataclasses import replace
    from pathlib import Path

    import httpx
    import yaml

    from nooa import connect
    from nooa.paths import get_user_dir

    from . import _connect_view as view
    from ._connect_prompts import confirm, environment_names, prompt

    async def show_checks(events):
        async with aclosing(events) as steps:
            async for event in steps:
                if isinstance(event, (connect.ConnectResult, connect.InterfaceResult)):
                    return event
                outcome = event.outcome
                if outcome["outcome"] == "running":
                    click.echo(f"Checking {event.name}...")
                else:
                    detail = outcome.get("error") or outcome.get("reason")
                    click.echo(
                        f"{event.name}: {outcome['outcome']}" + (f" ({detail})" if detail else "")
                    )
                    if outcome.get("reasoning_observed"):
                        click.echo(
                            "  Reasoning observed; acceptance alone does not prove a setting was obeyed."
                        )
        raise click.ClickException("Checks ended without a result.")

    path = Path(output) if output else get_user_dir("llm_config.yaml")
    try:
        default_style = "chat"
        approval = "none" if no_probe else probe
        interfaces = None
        view.intro(
            checks=approval != "none", output_tokens=output_tokens, budget_tokens=budget_tokens
        )
        view.step(1, "Connection")
        if provider and provider not in (*connect.PROVIDERS, "custom"):
            raise click.UsageError(
                "Unknown provider. Choose nvidia, openai, anthropic, google, openrouter, or custom."
            )
        if not provider and not endpoint and not yes:
            provider = prompt(
                "Choose a provider",
                choices=(*connect.PROVIDERS, "custom"),
                labels={
                    **{name: preset.label for name, preset in connect.PROVIDERS.items()},
                    "custom": "Custom endpoint",
                },
                open_menu=True,
            )
        if provider and provider != "custom":
            preset = connect.PROVIDERS[provider]
            endpoint = endpoint or preset.api_base
            default_style = preset.api_style
            if api_key_env is None:
                api_key_env = preset.api_key_env
        if yes and provider and provider != "custom":
            api_style = api_style or default_style
        if yes and not all((model, alias, endpoint, api_style)):
            raise click.UsageError("With --yes supply MODEL, --endpoint, --api-style and --as.")
        endpoint = endpoint or prompt(
            "Model server URL", suggestions=[p.api_base for p in connect.PROVIDERS.values()]
        )
        endpoint = connect.normalize_endpoint(endpoint)
        # Listing/authentication conventions do not choose the selected model's
        # generation interface. A mixed server can list all models via /models.
        discovery_style = api_style or default_style
        if api_key_env is None:
            default_env = (
                "ANTHROPIC_API_KEY" if discovery_style == "anthropic" else "OPENAI_API_KEY"
            )
            api_key_env = (
                default_env
                if yes
                else prompt(
                    "Key environment variable (enter - for no authentication)",
                    default=default_env,
                    suggestions=environment_names(
                        [p.api_key_env for p in connect.PROVIDERS.values()] + ["-"]
                    ),
                )
            )
            if api_key_env == "-":
                api_key_env = ""
        # Validate before using an endpoint or collecting a credential.
        connect.plan(
            alias or "candidate", model or "candidate", discovery_style, endpoint, api_key_env
        )
        needs_key = (
            not yes and approval != "none" and api_key_env and not os.environ.get(api_key_env)
        )
        api_key = (
            prompt("API key (used only for this setup)", hide_input=True)
            if prompt_key or needs_key
            else os.environ.get(api_key_env)
            if api_key_env
            else None
        )
        view.step(2, "Model")
        if not model:
            click.echo("Connecting to the server and listing models...")
            try:
                found = asyncio.run(
                    connect.discover(endpoint, api_style=discovery_style, api_key=api_key)
                )
            except connect.DiscoveryError as exc:
                click.echo(f"Could not list models: {exc}", err=True)
                if exc.status_code in {401, 403}:
                    raise click.ClickException(
                        "Authentication failed. Check the key and try again."
                    ) from None
                model = prompt("Exact model ID (if known; Ctrl-C to cancel)")
            else:
                endpoint = found.api_base
                names = [item["id"] for item in found.models]
                click.echo(
                    f"Connected. Found {len(names)} model(s). Type part of a name to search, then Tab to select."
                )
                model = prompt("Model", choices=names)
        view.step(3, "Connection checks")
        if not api_style:
            available = ("chat", "responses", "anthropic")
            if approval != "none":
                interfaces = asyncio.run(
                    show_checks(
                        connect.check_interfaces(
                            alias or "candidate",
                            model,
                            endpoint,
                            api_key_env,
                            budget_tokens=budget_tokens
                            if budget_tokens is not None
                            else 3 * (output_tokens + 512),
                            output_tokens=output_tokens,
                            api_key=api_key,
                        )
                    )
                )
                available = interfaces.accepted
                if not available:
                    raise click.ClickException(
                        "Could not confirm any interface. This is not proof they are unsupported. "
                        "Check credentials, the endpoint or the budget; use --api-style with --no-probe for manual setup."
                    )
                click.echo(
                    "Interfaces that returned the expected response format: " + ", ".join(available)
                )
            if interfaces and len(available) == 1:
                api_style = available[0]
                click.echo(f"Using {api_style} for {model}.")
            else:
                click.echo(f"Choose the request interface for {model}:")
                click.echo(
                    "chat = OpenAI-compatible; responses = OpenAI Responses; anthropic = Anthropic Messages."
                )
                api_style = prompt(
                    "API format",
                    choices=available,
                    default=default_style if default_style in available else available[0],
                )
        existing = None
        data = {}
        if path.exists():
            text = path.read_text()
            data = yaml.safe_load(text) or {}
            if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
                raise click.ClickException("Registry must contain a models mapping.")
        alias = alias or prompt(
            "Save this model as",
            default=model.rsplit("/", 1)[-1],
            suggestions=list(data.get("models", {})) + [model.rsplit("/", 1)[-1]],
            existing=tuple(data.get("models", {})),
        )
        if path.exists():
            existing = data.get("models", {}).get(alias)
            if alias in data.get("models", {}):
                click.echo(f"Warning: saving will overwrite model {alias!r} in {path}.", err=True)
                if not yes and not confirm("Replace this model?", default=False):
                    return
        candidate = None
        if no_catalogue and catalogue_model:
            raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
        if not no_catalogue:
            click.echo("Looking up public model information...")
            try:
                models = asyncio.run(connect.catalogue())
            except (httpx.HTTPError, ValueError, KeyError):
                if catalogue_model:
                    raise click.ClickException(
                        "Could not load the requested catalogue entry."
                    ) from None
                click.echo(
                    "Public catalogue unavailable; continuing with unknown limits.", err=True
                )
                models = []
            matches = (
                [item for item in models if item.get("id") == catalogue_model]
                if catalogue_model
                else connect.match_models(model, models)
            )
            if catalogue_model and not matches:
                raise click.ClickException("Requested catalogue model was not found.")
            if len(matches) == 1:
                view.model_details(matches[0], output_tokens=output_tokens)
                if yes or confirm("Use these model details?", default=True):
                    candidate = matches[0]
            elif matches:
                click.echo(
                    "Possible catalogue models: " + ", ".join(item["id"] for item in matches)
                )
                if yes:
                    raise click.ClickException(
                        "Ambiguous match: choose --catalogue-model or --no-catalogue."
                    )
                selected = prompt(
                    "Catalogue model (blank leaves it unknown)",
                    default="",
                    show_default=False,
                    choices=[""] + [item["id"] for item in matches],
                )
                if selected:
                    candidate = next((item for item in matches if item["id"] == selected), None)
                    if candidate is None:
                        raise click.ClickException("Choose one of the displayed model IDs.")
                    view.model_details(candidate, output_tokens=output_tokens)
                    if not confirm("Use these model details?", default=True):
                        candidate = None
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
            budget_tokens=budget_tokens if budget_tokens is not None else 4096,
            output_tokens=output_tokens,
            existing_entry=interfaces.results[api_style].entry if interfaces else existing,
        )
        interface_spent = interfaces.tokens_charged_to_budget if interfaces else 0
        # Once the model is selected we know how many levels need checking.
        # An explicit user limit stays shared and is never increased.
        remaining_estimate = sum(
            p.token_estimate
            for p in proposal.probes
            if (approval == "all" or approval == "minimal" and p.name == "routing")
            and not (
                proposal.entry["provenance"]["probes"].get(p.name, {}).get("outcome") == "accepted"
                and proposal.entry["provenance"]["probes"][p.name].get("request") == p.body
            )
        )
        proposal = replace(
            proposal,
            budget_tokens=remaining_estimate
            if budget_tokens is None
            else max(0, budget_tokens - interface_spent),
        )
        if interfaces:
            proposal.entry["provenance"]["interfaces"] = {
                style: result.entry["provenance"]["probes"]["routing"]
                for style, result in interfaces.results.items()
            }
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
            f"Estimated tokens for remaining checks: {remaining_estimate}; remaining budget: {proposal.budget_tokens}; price for the full plan: {price}."
        )
        if remaining_estimate > proposal.budget_tokens:
            click.echo(
                "Warning: --budget-tokens is too small for all checks. Some will be skipped. Increase it or omit it to check every level.",
                err=True,
            )
        click.echo(
            "No retries or capacity probes. Estimates are not billing limits: endpoints can ignore output caps."
        )
        result = asyncio.run(
            show_checks(connect.run_steps(proposal, approved=approval, api_key=api_key))
        )
        result.entry["provenance"]["tokens_charged_to_budget"] = (
            result.entry["provenance"].get("tokens_charged_to_budget", 0) + interface_spent
        )
        skipped = [
            name
            for name, record in result.entry["provenance"]["probes"].items()
            if record.get("reason") == "budget exhausted"
        ]
        if skipped:
            click.echo(
                "Warning: setup is incomplete; budget exhausted before "
                + ", ".join(skipped)
                + ". These settings have not been checked.",
                err=True,
            )
        view.step(4, "Save model")
        if yes or confirm(f"Write model entry to {path}?", default=True):
            connect.write(result.entry, path, alias=alias)
            click.echo(f"Saved {alias} to {path}.")
            if api_key and not os.environ.get(api_key_env):
                click.echo(
                    f"The key was not saved. Set {api_key_env} (or add it to your NOOA secrets file) before using this alias."
                )
            click.echo(f'Use it in Python: get_llm_client("{alias}")')
            if output:
                click.echo(
                    "For a custom path, include it in NEMO_OO_LLM_CONFIG or reload_registry(path)."
                )
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError) as exc:
        raise click.ClickException(str(exc)) from None
