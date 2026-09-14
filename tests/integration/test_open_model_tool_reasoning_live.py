# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hub reasoning/tool replay: two capped calls per model, no retries.

NOOA_RUN_OPEN_MODEL_REPLAY=1 uv run --env-file ../.env pytest -m integration -s
    tests/integration/test_open_model_tool_reasoning_live.py

Tests raw reasoning_content on the next HTTP request after SQLite close/reopen,
not merely its visibility somewhere in answer text. No opaque state is printed.
"""

import json
import os

import httpx
import pytest

from nooa.llm_types import LLMResponse
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient, Tool
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.retry_config import RetryConfig

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_OPEN_MODEL_REPLAY") != "1",
        reason="set NOOA_RUN_OPEN_MODEL_REPLAY=1 to spend inference tokens",
    ),
]

MODELS = {
    "deepseek": "openai/nvidia/deepseek-ai/deepseek-v4-pro",
    "kimi": "openai/nvidia/moonshotai/kimi-k3",
    "glm": "openai/nvidia/zai-org/glm-5.3",
    "qwen": "openai/nvidia/qwen/qwen3-5-397b-a17b",
}


def lookup(key: str) -> int:
    """Look up the offset needed to finish the comparison."""
    return 0


@pytest.mark.asyncio
@pytest.mark.parametrize("family", MODELS)
async def test_open_model_tool_reasoning_after_sqlite_resume(family, tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    sent = []
    original_send = httpx.AsyncClient.send

    async def capture(client, request, *args, **kwargs):
        if request.method == "POST" and request.url.host == "inference-api.nvidia.com":
            sent.append(json.loads(request.content))
        return await original_send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", capture)
    messages = [
        {"role": "system", "content": "Use the lookup tool when asked. Think briefly."},
        {
            "role": "user",
            "content": (
                "Is 17 times 19 less than 18 squared plus offset? First call lookup with "
                "key=offset; do not answer until the tool returns. Then give the difference."
            ),
        },
        {"role": "user", "content": "Live context: phase=before lookup."},
    ]
    options = {
        "model": MODELS[family],
        "api_base": "https://inference-api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_INFERENCE_API_KEY"],
        "max_tokens": 1536,
        "http_config": HttpConfig(read_timeout=120),
        "num_retries": 0,
        "retry_config": RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    }
    tools = [Tool(name="lookup", description="Look up an offset", callable=lookup)]
    async with CompletionClient(**options) as client:
        seed = await client.acall(messages, tools=tools)
    print(
        json.dumps(
            {
                "family": family,
                "phase": "seed",
                "usage": seed.usage.model_dump(),
                "finish_reason": seed.finish_reason,
                "reasoning_chars": len(seed.reasoning or ""),
            }
        ),
        flush=True,
    )
    raw = seed.raw_response.choices[0].message.reasoning_content
    assert isinstance(raw, str) and raw, "route returned no readable reasoning"
    assert seed.tool_calls and seed.finish_reason != "length"
    database = tmp_path / "session.db"
    with SQLiteStorageManager(database) as storage:
        storage.event_backend.store("1", seed)
    with SQLiteStorageManager(database) as storage:
        restored = next(storage.event_backend.all_events())
    assert isinstance(restored, LLMResponse)
    assert restored.reasoning == raw
    assert restored.raw_response is None
    history = [
        *messages,
        restored,
        *[
            {"role": "tool", "tool_call_id": call.id, "content": "0"}
            for call in restored.tool_calls
        ],
        {"role": "user", "content": "Live context: lookup complete. Answer concisely."},
    ]
    async with CompletionClient(**options) as client:
        result = await client.acall(history, tools=tools)
    assert len(sent) == 2, "probe must not retry"
    replay = next(m for m in sent[1]["messages"] if m.get("role") == "assistant")
    assert replay.get("reasoning_content") == raw, "native reasoning field changed on wire"
    assert result.finish_reason == "stop"
    assert result.content
    print(
        json.dumps(
            {
                "family": family,
                "phase": "resumed",
                "usage": result.usage.model_dump(),
                "wire_reasoning_equal": True,
                "sqlite_reasoning_equal": True,
            }
        ),
        flush=True,
    )
