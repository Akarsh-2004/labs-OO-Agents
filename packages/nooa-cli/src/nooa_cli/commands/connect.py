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
    help="Shared estimated-token budget for all checks (default: 65536); never increased after approval.",
)
@click.option("--output-tokens", type=click.IntRange(1, 4096), default=200, show_default=True)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Registry path; defaults to the user llm_config.yaml.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Approve the checks and save without prompting; supply the connection options.",
)
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
    from ._connect_prompts import confirm, edit_model_details, environment_names, prompt

    async def show_checks(events):
        with view.quiet_provider_messages():
            return await display_checks(events)

    async def display_checks(events):
        async with aclosing(events) as steps:
            async for event in steps:
                if isinstance(event, (connect.ConnectResult, connect.InterfaceResult)):
                    return event
                outcome = event.outcome
                if outcome["outcome"] == "running":
                    click.echo(f"Checking {event.name}...")
                else:
                    detail = (
                        view.check_failure(outcome) or outcome.get("error") or outcome.get("reason")
                    )
                    label = "not confirmed" if outcome.get("error") else outcome["outcome"]
                    click.echo(f"{event.name}: {label}" + (f" ({detail})" if detail else ""))
                    if outcome.get("reasoning_observed"):
                        click.echo("  Reasoning included in the response.")
                    if event.name == "cache" and outcome.get("input_tokens"):
                        cached = outcome.get("cached_input_tokens") or 0
                        total = outcome["input_tokens"]
                        click.echo(
                            f"  Reused {cached:,} / {total:,} input tokens ({cached / total:.0%})."
                        )
        raise click.ClickException("Checks ended without a result.")

    path = Path(output) if output else get_user_dir("llm_config.yaml")
    try:
        data = yaml.safe_load(path.read_text()) or {} if path.exists() else {}
        if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
            raise click.ClickException("Registry must contain a models mapping.")
        server_urls = [p.api_base for p in connect.PROVIDERS.values()]
        for entry in data.get("models", {}).values():
            address = entry.get("api_base") if isinstance(entry, dict) else None
            if isinstance(address, str):
                try:
                    server_urls.append(connect.normalize_endpoint(address))
                except ValueError:
                    pass  # Do not offer malformed URLs or embedded credentials.
        server_urls = list(dict.fromkeys(server_urls))
        default_style = "chat"
        approval = "none" if no_probe else probe
        budget_tokens = 65536 if budget_tokens is None else budget_tokens
        interfaces = None
        view.intro(
            checks=approval != "none", output_tokens=output_tokens, budget_tokens=budget_tokens
        )
        if approval != "none" and not yes:
            if not confirm("Approve API checks within this budget?", default=False):
                click.echo("No API checks approved. Run with --no-probe for manual setup.")
                return
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
        endpoint = endpoint or prompt("Model server URL", suggestions=server_urls, open_menu=True)
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
                    f"Server listed {len(names)} model(s). Credentials are checked next. Type part of a name to search, then Tab to select."
                )
                model = prompt("Model", choices=names)
        view.step(3, "Connection checks")
        if api_style == "responses":
            view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
        if not api_style:
            available = ("chat", "responses", "anthropic")
            if approval != "none":
                interface_spent = 0
                while True:
                    interfaces = asyncio.run(
                        show_checks(
                            connect.check_interfaces(
                                alias or "candidate",
                                model,
                                endpoint,
                                api_key_env,
                                budget_tokens=max(0, budget_tokens - interface_spent),
                                output_tokens=output_tokens,
                                api_key=api_key,
                            )
                        )
                    )
                    interface_spent += interfaces.tokens_charged_to_budget
                    interfaces = replace(interfaces, tokens_charged_to_budget=interface_spent)
                    available = interfaces.accepted
                    if available:
                        break
                    view.line(
                        "Could not confirm a working connection. Listing models does not validate the key.",
                        fg="yellow",
                    )
                    if yes:
                        raise click.ClickException(
                            "Check credentials and endpoint, or run without --yes to correct them interactively."
                        )
                    remaining = max(0, budget_tokens - interface_spent)
                    if remaining < output_tokens + 512:
                        view.line(
                            "The approved check budget is exhausted. Nothing was saved; restart setup to approve a new budget."
                        )
                        return
                    view.line(
                        f"You can correct the connection here. {remaining:,} estimated tokens remain in the approved budget."
                    )
                    action = prompt(
                        "Next step",
                        choices=("key", "server", "retry", "cancel"),
                        default="key",
                        labels={
                            "key": "Change key",
                            "server": "Edit server and model",
                            "retry": "Try again unchanged",
                            "cancel": "Exit without saving",
                        },
                    )
                    if action == "cancel":
                        click.echo("Setup cancelled. Nothing was saved.")
                        return
                    if action == "key":
                        source = prompt(
                            "Key environment variable (or paste for a temporary key)",
                            default=api_key_env or "paste",
                            suggestions=environment_names(["paste"]),
                        )
                        if source == "paste":
                            api_key = prompt("API key (used only for this setup)", hide_input=True)
                        else:
                            api_key_env = source
                            api_key = os.environ.get(source)
                            if not api_key:
                                view.line(
                                    "That variable is unset or empty. You can paste a temporary key instead."
                                )
                                api_key = prompt(
                                    "API key (used only for this setup)", hide_input=True
                                )
                    elif action == "server":
                        endpoint = connect.normalize_endpoint(
                            prompt(
                                "Model server URL",
                                default=endpoint,
                                suggestions=server_urls,
                                open_menu=True,
                            )
                        )
                        model = prompt("Exact model ID", default=model)
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
            if api_style == "responses":
                view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
        existing = None
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
        edited_settings = False
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
            else:
                click.echo("No catalogue match; model limits and reasoning levels remain unknown.")
        if candidate is not None:
            while True:
                view.model_details(candidate, output_tokens=output_tokens, edited=edited_settings)
                action = (
                    "use"
                    if yes
                    else prompt(
                        "Model settings",
                        default="use",
                        choices=("use", "edit", "skip", "cancel"),
                        labels={
                            "use": "Use these settings",
                            "edit": "Edit settings",
                            "skip": "Continue without these settings",
                            "cancel": "Cancel setup",
                        },
                        open_menu=True,
                    )
                )
                if action == "cancel":
                    click.echo("Setup cancelled. Nothing saved.")
                    return
                if action == "skip":
                    candidate = None
                    edited_settings = False
                    click.echo(
                        "Continuing without the published model settings. Explicit command-line settings still apply."
                    )
                    break
                if action == "use":
                    break
                candidate = edit_model_details(candidate)
                edited_settings = True
            if edited_settings:
                # Apply edits only after confirmation, not if the user skips them.
                context_window = None
                levels_file = levels = reasoning_template = None
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
            session_checks=approval == "all",
        )
        if edited_settings:
            for field in (
                "context_window",
                "max_output_tokens",
                "reasoning_levels",
                "reasoning_default",
            ):
                proposal.entry["provenance"][field] = {
                    "source": "user",
                    "value": proposal.entry.get(field),
                }
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
        if proposal.session_checks:
            from nooa._connect_session import TOKEN_RESERVATION

            remaining_estimate += TOKEN_RESERVATION
        proposal = replace(
            proposal,
            budget_tokens=max(0, budget_tokens - interface_spent),
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
                "No context window selected. The runtime will use its fallback; set --context-window to supply a limit."
            )
        click.echo(yaml.safe_dump({"models": {alias: proposal.entry}}, sort_keys=False))
        price = (
            "unknown"
            if proposal.price_estimate is None
            else f"~${proposal.price_estimate:.6f} at catalogue prices"
        )
        click.echo(
            f"Plan: {len(proposal.probes) + (3 if proposal.session_checks else 0)} candidate calls; basic checks {output_tokens} output tokens per call, session checks 2048; approval: {approval}."
        )
        click.echo(
            f"Estimated tokens for remaining checks: {remaining_estimate}; remaining budget: {proposal.budget_tokens}; price for the full plan: {price}."
        )
        if remaining_estimate > proposal.budget_tokens:
            click.echo(
                "Warning: the approved budget is too small for all checks. Some will be skipped. Restart with a larger --budget-tokens value to run them all.",
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
        unobserved = connect.unobserved_reasoning_levels(result.entry)
        if unobserved:
            click.echo(
                "Warning: no reasoning information was returned for: "
                + ", ".join(unobserved)
                + ". These checks have not confirmed reasoning for those levels. "
                "Try another API format or review the server's reasoning settings. "
                "Some servers do not expose reasoning information.",
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
