# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded multi-turn onboarding checks; raw replay state stays in memory."""

import asyncio
import json
from copy import deepcopy

REPLY_CAP = 2048
# Includes padding, schema/instructions, and up to two prior reply-sized items.
CALL_RESERVATION = 8192 + 3 * REPLY_CAP
TOKEN_RESERVATION = 3 * CALL_RESERVATION


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _contains(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(value, actual[key]) for key, value in expected.items()
        )
    return expected == actual


def _wire_reasoning(value):
    """Only replay fields count; portable text is not native reasoning replay."""
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {
                "reasoning_content",
                "encrypted_content",
                "signature",
                "thought_signature",
                "thought_signatures",
            }:
                yield from _strings(child)
            elif key == "thinking" and value.get("type") == "thinking":
                yield from _strings(child)
            elif key == "data" and value.get("type") == "redacted_thinking":
                yield from _strings(child)
            elif key == "summary" and value.get("type") == "reasoning":
                yield from _strings(child)
            elif key == "id" and isinstance(child, str) and "__thought__" in child:
                yield child.split("__thought__", 1)[1]
            else:
                yield from _wire_reasoning(child)
    elif isinstance(value, list):
        for child in value:
            yield from _wire_reasoning(child)


def _cache_markers(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"cache_control", "prompt_cache_breakpoint"}:
                yield key
            else:
                yield from _cache_markers(child)
    elif isinstance(value, list):
        for child in value:
            yield from _cache_markers(child)


def _reasoning_values(response):
    """Compare meaningful reasoning payloads, never persist their contents."""
    from nooa._immutable_json import json_containers

    def payloads(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {
                    "encrypted_content",
                    "signature",
                    "data",
                    "thought_signature",
                    "thought_signatures",
                    "inline_thought_signature",
                }:
                    yield from _strings(child)
                else:
                    yield from payloads(child)
        elif isinstance(value, list):
            for child in value:
                yield from payloads(child)

    for part in response.parts:
        if part.kind == "reasoning" and part.text:
            yield part.text
        native = json_containers(part.native) if part.native else {}
        yield from payloads(native)


async def session_steps(alias, entry, *, api_key, budget_tokens):
    """Three calls on one client; no retry, invented reasoning, or tool execution."""
    from nooa.connect import ProbeUpdate, _include_rejected
    from nooa.context_blocks.formatter import OpenAIProviderFormatter, ResponsesProviderFormatter
    from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
    from nooa.context_blocks.renderer import render_context
    from nooa.context_blocks.renderers.cached import CachedBlockFormatter
    from nooa.unifiedllm import CacheBoundary, RetryConfig, Tool
    from nooa.unifiedllm.registry import client_from_config

    if budget_tokens < TOKEN_RESERVATION:
        yield ProbeUpdate("session", {"outcome": "not_probed", "reason": "budget exhausted"})
        return
    window = entry.get("context_window")
    if isinstance(window, int) and window < CALL_RESERVATION:
        yield ProbeUpdate(
            "session",
            {"outcome": "not_probed", "reason": "context window too small for this check"},
        )
        return
    levels = entry.get("reasoning_levels", {})
    level = entry.get("reasoning_default")
    if level not in levels:
        level = next(iter(levels), None)
    settings = levels.get(level, {})
    controls = {
        key: value
        for key, value in {**entry, **settings}.items()
        if key
        in {"thinking", "reasoning", "reasoning_effort", "output_config", "chat_template_kwargs"}
    }
    if any(
        settings.get(key, REPLY_CAP) != REPLY_CAP
        for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
    ):
        yield ProbeUpdate(
            "session",
            {
                "outcome": "not_probed",
                "reason": "selected reasoning level changes the approved reply cap",
            },
        )
        return

    def probe_tool(value: str):
        raise RuntimeError("Setup checks never execute model tools")

    formatter = CachedBlockFormatter()
    provider = (
        ResponsesProviderFormatter()
        if entry["api_style"] == "responses"
        else OpenAIProviderFormatter()
    )
    # Non-repetitive lines give a reusable prefix without any user's private data.
    padding = "\n".join(
        f"Reference record {i:04d}: item {i * 17 + 3:06d} belongs to batch {i % 97:02d}."
        for i in range(360)
    )
    messages = render_context(
        [
            ResolvedBlock(
                key="instructions",
                role=Role.SYSTEM,
                content="Use the reference to solve the task. Do not repeat the reference.",
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="reference",
                role=Role.USER,
                content=padding,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="task",
                role=Role.USER,
                content="Find the sum of item numbers for records 0013 and 0027. Call probe_tool with the sum as a string, or answer briefly.",
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ],
        block_formatter=formatter,
        provider_formatter=provider,
    ).output
    params = {
        "tools": [
            Tool(name="probe_tool", description="Record a computed value", callable=probe_tool)
        ],
        "max_output_tokens" if entry["api_style"] == "responses" else "max_tokens": REPLY_CAP,
    }
    if level is not None:
        params["reasoning_level"] = level
    bodies = []

    async def capture(request):
        bodies.append(json.loads(request.content))

    try:
        client = client_from_config(
            alias,
            entry,
            api_key=api_key,
            num_retries=0,
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        )
    except Exception as exc:
        yield ProbeUpdate("session", {"outcome": "not_confirmed", "error": type(exc).__name__})
        return
    # Hook only this owned client. Never monkey-patch global HTTP or store headers.
    hooks = client._http.httpx_async.event_hooks["request"]
    hooks.append(capture)
    spent = 0
    first = None
    replay = None
    last_usage = None
    observations = []
    settings_ok = True
    try:
        for index, name in enumerate(("seed", "replay", "repeat")):
            if spent + CALL_RESERVATION > budget_tokens:
                yield ProbeUpdate(
                    "session",
                    {
                        "outcome": "not_probed",
                        "reason": "budget exhausted",
                        "tokens_charged_to_budget": spent,
                    },
                )
                return
            yield ProbeUpdate(f"session:{name}", {"outcome": "running"})
            call_messages = (
                messages
                if index == 0
                else [
                    *replay,
                    CacheBoundary(),
                    {
                        "role": "user",
                        "content": f"Check {index}: verify the recorded result briefly.",
                    },
                ]
            )
            before = len(bodies)
            spent += CALL_RESERVATION
            try:
                async with asyncio.timeout(120):
                    response = await client.acall(call_messages, **params)
            except Exception as exc:
                record = {
                    "outcome": "not_confirmed",
                    "error": type(exc).__name__,
                    "tokens_charged_to_budget": spent,
                }
                status = getattr(exc, "status_code", None)
                if isinstance(status, int):
                    record["status_code"] = status
                if entry.get("include") == ["reasoning.encrypted_content"] and _include_rejected(
                    exc
                ):
                    record["include_rejected"] = True
                yield ProbeUpdate("session", record)
                return
            usage = response.usage
            last_usage = usage
            total = usage.input_tokens + usage.output_tokens if usage else 0
            spent += max(0, total - CALL_RESERVATION)
            observed = bool(
                any(p.kind == "reasoning" for p in response.parts)
                or (usage and usage.reasoning_tokens)
            )
            observations.append(observed)
            record = {
                "outcome": "accepted",
                "reasoning_observed": observed,
                "input_tokens": usage.input_tokens if usage else None,
                "output_tokens": usage.output_tokens if usage else None,
                "cached_input_tokens": usage.cached_input_tokens if usage else None,
                "finish_reason": response.finish_reason,
                "tested_reasoning_level": level,
            }
            yield ProbeUpdate(f"session:{name}", record)
            if (
                len(bodies) != before + 1
                or (usage and usage.output_tokens > REPLY_CAP)
                or response.finish_reason in {"length", "error"}
            ):
                yield ProbeUpdate(
                    "session",
                    {
                        "outcome": "not_confirmed",
                        "reason": "request capture missing, reply truncated, or cap exceeded",
                        "tokens_charged_to_budget": spent,
                    },
                )
                return
            wire = bodies[-1]
            settings_ok &= _contains(controls, wire)
            if index == 0:
                first = response
                replay = [m for m in messages if not isinstance(m, CacheBoundary)] + [first]
                for call in first.tool_calls:
                    if call.name != "probe_tool":
                        yield ProbeUpdate(
                            "session",
                            {
                                "outcome": "not_confirmed",
                                "reason": "unexpected tool requested",
                                "tokens_charged_to_budget": spent,
                            },
                        )
                        return
                    replay.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": "Probe result recorded.",
                        }
                    )

        # Compare all provider fields; only the final user message may differ.
        left, right = deepcopy(bodies[1]), deepcopy(bodies[2])
        key = "input" if entry["api_style"] == "responses" else "messages"
        for body in (left, right):
            if entry["api_style"] == "anthropic":
                body[key][-1]["content"][-1]["text"] = "<volatile>"
            else:
                body[key][-1]["content"] = "<volatile>"
        stable = left == right
        cached = last_usage.cached_input_tokens if last_usage else None
        seed_input = first.usage.input_tokens if first.usage else 0
        markers = list(_cache_markers(bodies[2]))
        explicit = bool(markers)
        substantial = bool(stable and seed_input > 0 and cached and cached >= seed_input / 2)
        cache_reason = (
            "Reusable conversation cached"
            if substantial
            else "Only a small portion was cached; try a longer prompt or check the server's caching support"
            if cached
            else "Cache markers were sent, but the server reported no reuse; retry later or check server support"
            if explicit
            else "No cache markers or cache reads observed; check the runtime's cache defaults and server support"
        )
        yield ProbeUpdate(
            "cache",
            {
                "outcome": "confirmed" if substantial else "not_confirmed",
                "cached_input_tokens": cached,
                "input_tokens": last_usage.input_tokens if last_usage else None,
                "stable_prefix": stable,
                "marker_count": len(markers),
                "explicit_mode": bodies[2].get("prompt_cache_options", {}).get("mode")
                == "explicit",
                "reason": cache_reason,
            },
        )
        expected = list(_reasoning_values(first))
        actual = list(_wire_reasoning(bodies[1]))
        retained = bool(expected) and all(value in actual for value in expected)
        yield ProbeUpdate(
            "reasoning_retention",
            {
                "outcome": "confirmed" if retained and settings_ok else "not_confirmed",
                "reasoning_observed_by_turn": observations,
                "settings_retained": settings_ok if controls else None,
                "state_retained": retained if expected else None,
                "tested_reasoning_level": level,
                "reason": "Reasoning preserved across turns"
                if retained and settings_ok
                else "Reasoning was returned but not preserved in the follow-up request; check replay settings or try another interface"
                if expected
                else "No replayable reasoning returned; check reasoning/replay settings or try another interface. This does not mean reasoning is off",
            },
        )
        yield ProbeUpdate("session", {"outcome": "completed", "tokens_charged_to_budget": spent})
    finally:
        hooks.remove(capture)
        await client.aclose()
