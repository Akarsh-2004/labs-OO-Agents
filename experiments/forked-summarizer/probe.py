# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Six bounded Hub calls comparing forked and standalone summary cache reads."""

import asyncio
import json
from uuid import uuid4

from nooa.context_blocks import BlockMetadata, ResolvedBlock, Role, render_context
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.unifiedllm import Tool

# Reuse the existing opt-in probe's Hub clients; no new routing configuration.
from tests.integration.test_cache_resume_live import _client


def return_result(result: str) -> str:
    raise AssertionError("This experiment must never execute model tools")


async def main():
    for family in ("openai", "anthropic"):
        notes = "\n".join(
            f"Record {i}: amber birch cedar dune elm fern grove hill." for i in range(400)
        )
        notes += "\nDecision: launch on Tuesday. Budget: 42 units. Owner: Alex."
        blocks = [
            ResolvedBlock(
                key="instructions",
                role=Role.SYSTEM,
                content=f"Experiment {uuid4().hex}. Follow the latest request. Keep answers short.",
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(key="notes", role=Role.USER, content=notes),
            ResolvedBlock(
                key="live",
                role=Role.SYSTEM,
                content="phase=parent; pending=summary",
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ]
        messages = render_context(
            blocks,
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        suffix = {
            "role": "user",
            "content": (
                "Summarize the decision, budget and owner above in under 250 characters. "
                "Write only the summary as plain text; do not call tools."
            ),
        }
        params = {
            "tools": [
                Tool(name="return_result", description="Return a result", callable=return_result)
            ],
            "tool_choice": "auto",
            "prompt_cache_key": f"summary-probe-{uuid4().hex}",
        }
        async with _client(family) as client:
            for label, request in (
                ("parent", [*messages, {"role": "user", "content": "Reply READY."}]),
                ("fork", [*messages, suffix]),
                (
                    "standalone",
                    [
                        {"role": "system", "content": f"Independent summarizer {uuid4().hex}."},
                        {"role": "user", "content": notes + "\n" + suffix["content"]},
                    ],
                ),
            ):
                response = await client.acall(request, **params)
                usage = response.usage
                assert usage is not None
                print(
                    json.dumps(
                        {
                            "provider": family,
                            "phase": label,
                            "usage": usage.model_dump(),
                            "text_chars": len(response.content),
                            "calls": len(response.tool_calls),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    asyncio.run(main())
