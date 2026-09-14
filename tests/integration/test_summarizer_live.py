# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in installed-agent summarization: three capped requests per provider.

Set NOOA_RUN_SUMMARIZER_E2E=1 and install release-gate-openai/anthropic registry
aliases. Routes and credentials belong to that configuration, not this test.
Reports contain only counts; captured requests remain in memory.
"""

import asyncio
import json
import os
from urllib.parse import urlparse

import httpx
import pytest

from nooa import Agent, Context, strategy
from nooa.agents import TokenBudgetSummarizer
from nooa.config.strategy_config import CodeActConfig
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.context_blocks.events import UserEvent
from nooa.events import Summary
from nooa.strategies import CodeActStrategy
from nooa.strategies.codeact import return_text_as_result
from nooa.unifiedllm import HttpConfig, RetryConfig, registry

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_SUMMARIZER_E2E") != "1",
        reason="set NOOA_RUN_SUMMARIZER_E2E=1 to spend inference tokens",
    ),
]


class SummaryParent(Agent):
    @strategy(
        CodeActStrategy(
            config=CodeActConfig(max_iterations=1, max_retries=1),
            on_text_only=return_text_as_result,
        )
    )
    async def reply(self, request: str) -> str:
        """Answer the request from the stored notes. No research or Python is needed."""
        ...


async def exercise_summarization(client, family, monkeypatch):
    """Run real agent methods and require the next turn to apply the pending summary."""
    responses, bodies = [], []
    release = asyncio.Event()
    started = asyncio.Event()
    real_call, real_send = client.acall, httpx.AsyncClient.send
    host = urlparse(client.config["api_base"]).hostname
    call_count = 0

    async def capture(http_client, request, **kwargs):
        if request.url.host == host and request.method == "POST":
            assert len(bodies) < 3, "Provider request budget exceeded"
            bodies.append(json.loads(request.content))
        return await real_send(http_client, request, **kwargs)

    async def call(messages, **params):
        nonlocal call_count
        call_count += 1
        assert call_count <= 3, "Model-call budget exceeded"
        if call_count == 2:
            assert "Background memory compaction" in messages[-1]["content"]
            started.set()
            await release.wait()
        response = await real_call(messages, **params)
        responses.append(response)
        return response

    monkeypatch.setattr(httpx.AsyncClient, "send", capture)
    monkeypatch.setattr(client, "acall", call)
    parent = SummaryParent(
        llm=client, truncation=TruncationConfig(event_format=FormatConfig(max_string=None))
    )
    parent.context["live"] = Context(expr="'summary smoke test'")
    notes = UserEvent(
        content="\n".join(
            f"Record {i}: amber birch cedar dune elm fern grove hill." for i in range(400)
        )
        + "\nDecision: launch Tuesday. Budget: 42 units. Owner: Alex."
    )
    original_tag = parent.event_manager.add(notes)
    summarizer = TokenBudgetSummarizer.install(
        parent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1, target_chars=350)
    )
    try:
        assert "READY" in await parent.reply("Acknowledge the notes with READY.")
        assert summarizer._pending_task is not None, "Installed summarizer did not fork"
        await asyncio.wait_for(started.wait(), 5)
        assert not summarizer._pending_task.done(), "Parent must return before summary completes"
        assert not any(isinstance(e, Summary) for e in parent.event_manager.values())
        source = dict(summarizer._pending_source)
        recent = {tag: e.id for tag, e in parent.event_manager.items() if tag not in source}
        before_fork = [(tag, e.id) for tag, e in parent.event_manager.items()]
        release.set()
        await asyncio.wait_for(summarizer._pending_task, 150)
        text = summarizer._pending_summary
        assert [(tag, e.id) for tag, e in parent.event_manager.items()] == before_fork, (
            "Fork wrote parent events or executed tools"
        )
        assert text, "Background summary failed or returned unusable text"
        assert all(fact in text for fact in ("Tuesday", "42", "Alex")), "Summary lost a key fact"
        assert original_tag in parent.event_manager.keys(), "Summary applied before next turn"

        # Keep this a three-request smoke test; next turn applies the waiting
        # summary through BeforeTurn, but must not schedule another summary.
        summarizer.config = summarizer.config.model_copy(update={"max_tokens": 1_000_000_000})
        answer = await parent.reply("What are the launch day, budget and owner? Be concise.")
        summaries = [e for e in parent.event_manager.values() if isinstance(e, Summary)]
        assert len(summaries) == 1, "Next agent turn did not apply exactly one summary"
        summary = summaries[0]
        assert summary.summary_text == text
        assert original_tag in summary.children_tags
        assert original_tag not in parent.event_manager.keys()
        assert parent.events[original_tag].id == notes.id, "Raw source was not preserved"
        assert all(parent.event_manager[tag].id == identity for tag, identity in recent.items())
        assert all(fact in answer for fact in ("Tuesday", "42", "Alex")), "Continuation lost a fact"
        assert len(bodies) == len(responses) == call_count == 3

        first, fork, continuation = bodies
        key = "input" if family == "openai" else "messages"
        assert {k: v for k, v in first.items() if k != key} == {
            k: v for k, v in fork.items() if k != key
        }, "Fork changed request settings"
        if family == "openai":
            assert fork[key][:-1] == first[key], "Fork changed the parent prefix"
        else:
            assert fork[key][:-1] == first[key][:-1], "Fork changed the parent prefix"
            assert fork[key][-1]["content"][:-1] == first[key][-1]["content"]
        # Summary renderers may quote/escape multiline text. Check its facts in
        # the outgoing request as well as the exact stored summary above.
        assert all(fact in json.dumps(continuation) for fact in ("Tuesday", "42", "Alex")), (
            "Next request lost summary facts"
        )
        assert "Record 399: amber" not in json.dumps(continuation), "Archived notes still rendered"
        assert all(response.usage is not None for response in responses)
        return {
            "family": family,
            "requests": call_count,
            "applied": True,
            "calls": [
                {
                    "input_tokens": r.usage.input_tokens,
                    "output_tokens": r.usage.output_tokens,
                    "cached_input_tokens": r.usage.cached_input_tokens,
                }
                for r in responses
            ],
        }
    finally:
        release.set()
        await parent.aclose()


@pytest.mark.parametrize("family", ["openai", "anthropic"])
async def test_installed_summarizer_applies_before_next_turn(family, monkeypatch):
    """Require a generated, applied summary and a successful continuation on real providers."""
    registry.ensure_loaded()
    alias = f"release-gate-{family}"
    if alias not in registry.MODELS:
        pytest.skip(f"registry alias {alias!r} is not installed")
    limit = {"max_output_tokens": 2048} if family == "openai" else {"max_tokens": 2048}
    async with registry.get_llm_client(
        alias,
        **limit,
        num_retries=0,
        cache_breakpoint=family,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        http_config=HttpConfig(read_timeout=120),
    ) as client:
        report = await exercise_summarization(client, family, monkeypatch)
    print(json.dumps(report), flush=True)
    assert report["calls"][1]["cached_input_tokens"] > 0, "Summary did not reuse the parent cache"
