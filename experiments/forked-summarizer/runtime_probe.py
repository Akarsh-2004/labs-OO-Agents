# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four bounded calls exercising installed summarization on a real CodeAct agent."""

import asyncio
import json

from nooa import Agent, Context, strategy
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.context_blocks.events import UserEvent
from nooa.strategies import CodeActStrategy
from nooa.strategies.codeact import return_text_as_result
from tests.integration.test_cache_resume_live import _client


class Parent(Agent):
    @strategy(CodeActStrategy(on_text_only=return_text_as_result))
    async def ready(self) -> str:
        """Acknowledge the stored notes with READY. No research or code is needed."""
        ...


async def main():
    for family in ("openai", "anthropic"):
        async with _client(family) as client:
            calls = []
            original = client.acall

            async def record(messages, _original=original, _calls=calls, _family=family, **params):
                result = await _original(messages, **params)
                _calls.append(result)
                print(
                    json.dumps(
                        {
                            "family": _family,
                            "call": len(_calls),
                            "usage": result.usage.model_dump(),
                            "text_chars": len(result.content),
                            "tools": [call.name for call in result.tool_calls],
                        }
                    ),
                    flush=True,
                )
                return result

            client.acall = record
            parent = Parent(
                llm=client, truncation=TruncationConfig(event_format=FormatConfig(max_string=None))
            )
            parent.context["live"] = Context(expr="'phase=parent; pending=summary'")
            parent.event_manager.add(
                UserEvent(
                    content="\n".join(
                        f"Record {i}: amber birch cedar dune elm fern grove hill."
                        for i in range(400)
                    )
                    + "\nDecision: launch Tuesday. Budget: 42 units. Owner: Alex."
                )
            )
            summarizer = TokenBudgetSummarizer.install(
                parent,
                config=TokenBudgetConfig(max_tokens=100, preserve_recent=1, target_chars=350),
            )
            try:
                await parent.ready()
                assert summarizer._pending_task is not None
                await summarizer._pending_task
                assert summarizer._pending_summary, "The installed fork produced no summary"
                text = summarizer._pending_summary
                assert all(word in text for word in ("42", "Alex", "Tuesday")), (
                    "The summary omitted a required fact"
                )
                summarizer._apply_pending_summary()
                print(
                    json.dumps(
                        {
                            "family": family,
                            "applied": True,
                            "cached_fraction": calls[-1].usage.cached_input_tokens
                            / calls[-1].usage.input_tokens,
                        }
                    ),
                    flush=True,
                )
            finally:
                summarizer._uninstall()


if __name__ == "__main__":
    asyncio.run(main())
