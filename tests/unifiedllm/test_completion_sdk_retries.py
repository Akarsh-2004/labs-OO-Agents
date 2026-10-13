# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RetryConfig is the only endpoint retry budget for OpenAI-compatible chat routes.

LiteLLM defaults the OpenAI SDK to two retries of its own, so each NOOA attempt
became three HTTP requests. These tests count requests at the HTTP transport.
"""

import asyncio
from typing import Any

import httpx
import pytest
from litellm.llms.openai.openai import OpenAIChatCompletion

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.unifiedllm import _ClientHttp

OK_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
MESSAGES: list[Any] = [{"role": "user", "content": "hi"}]


def _retry(max_retries: int) -> RetryConfig:
    return RetryConfig(
        max_retries=max_retries, rate_limit_extra_retries=0, base_delay=0.0, jitter_factor=0.0
    )


def _install(monkeypatch, transport: Any) -> None:
    """Route NOOA's owned clients and LiteLLM's default pool to ``transport``."""
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_sync_http_client",
        staticmethod(lambda **kwargs: httpx.Client(transport=transport)),
    )
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_async_http_client",
        staticmethod(lambda **kwargs: httpx.AsyncClient(transport=transport)),
    )


def _client(nooa_retries: int, **config) -> CompletionClient:
    return CompletionClient(
        "openai/m",
        api_base="https://endpoint.example/v1",
        api_key="key",
        retry_config=_retry(nooa_retries),
        **config,
    )


async def _call(client: CompletionClient, is_async: bool, **kwargs):
    if is_async:
        return await client.acall(MESSAGES, **kwargs)
    return client.call(MESSAGES, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"api_key": "override-key"}],
    ids=["owned-client", "litellm-client"],
)
async def test_persistent_error_uses_exactly_the_configured_attempts(
    monkeypatch, is_async, max_retries, overrides
):
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(max_retries)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            await _call(client, is_async, **overrides)
    finally:
        await client.aclose()

    assert attempts == max_retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_owned_sdk_client_has_no_hidden_retries(monkeypatch, is_async):
    _install(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(200, json=OK_BODY)))
    client = _client(2)
    try:
        await _call(client, is_async)
        http = client._http
        assert http is not None
        sdk_client = http.async_client if is_async else http.sync_client
        assert sdk_client.max_retries == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_transient_error_recovers_on_nooa_retry(monkeypatch, is_async):
    statuses: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        status = 503 if not statuses else 200
        statuses.append(status)
        if status == 503:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json=OK_BODY)

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(2)
    try:
        response = await _call(client, is_async)
    finally:
        await client.aclose()

    assert response.content == "done"
    assert statuses == [503, 200]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_explicit_sdk_retries_remain_opt_in(monkeypatch, is_async):
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(0, max_retries=1)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            await _call(client, is_async)
    finally:
        await client.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_cancellation_stops_without_further_attempts(monkeypatch):
    attempts = 0
    started = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        started.set()
        await asyncio.sleep(60)
        return httpx.Response(200, json=OK_BODY)

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(2)
    try:
        task = asyncio.create_task(client.acall(MESSAGES))
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        # Neither NOOA nor the SDK may treat cancellation as a retryable failure.
        await asyncio.sleep(0.2)
    finally:
        await client.aclose()

    assert attempts == 1
