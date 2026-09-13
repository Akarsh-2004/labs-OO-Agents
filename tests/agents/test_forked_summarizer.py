# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A summary forks the sent request, never the running agent or its tools."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from nooa import Agent
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig
from nooa.events import Message
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CacheBoundary, FakeLLMClient, LLMResponse, LLMUsage, ToolCall


def response(text="summary", **kwargs):
    return LLMResponse(content=text, usage=LLMUsage(input_tokens=1000, output_tokens=5), **kwargs)


def setup():
    agent = Agent(llm=FakeLLMClient())
    for i in range(4):
        agent.event_manager.add(Message(content=f"fact {i}"))
    summarizer = TokenBudgetSummarizer.install(
        agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
    )
    turn = response("old answer")
    boundary = CacheBoundary()
    ctx = LLMCallContext(
        agent=agent,
        runtime=agent.runtime,
        client=agent.llm,
        messages=[
            {"role": "system", "content": "stable"},
            turn,
            boundary,
            {"role": "user", "content": [{"type": "text", "text": "live=1"}]},
        ],
        params={
            "tools": [],
            "output_model": None,
            "tool_choice": "auto",
            "prompt_cache_key": "parent-shard",
            "extra_body": {"setting": "original"},
        },
    )
    return agent, summarizer, ctx


@pytest.mark.asyncio
async def test_fork_is_background_isolated_and_applied_only_at_boundary():
    agent, summarizer, ctx = setup()
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    async def summary_call(messages, **params):
        seen.append((messages, params))
        entered.set()
        await release.wait()
        return response()

    agent.llm.acall = summary_call
    parent = response("parent answer")

    async def core(request):
        request.response = parent
        return request

    result = await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert result.response is parent
    assert summarizer._pending_task is not None
    await asyncio.wait_for(entered.wait(), 1)
    assert not summarizer._pending_task.done()
    assert agent.event_manager.keys() == ["1", "2", "3", "4"]
    assert seen[0][0][1] is ctx.messages[1]
    assert seen[0][0][2] is ctx.messages[2]
    assert seen[0][1] == ctx.params
    assert seen[0][0][-1]["role"] == "user"
    assert "1" in seen[0][0][-1]["content"] and "3" in seen[0][0][-1]["content"]
    ctx.messages[-1]["content"][0]["text"] = "changed"
    ctx.params["extra_body"]["setting"] = "changed"
    assert seen[0][0][-2]["content"][0]["text"] == "live=1"
    assert seen[0][1]["extra_body"]["setting"] == "original"
    agent.event_manager.add(Message(content="new work"))
    release.set()
    await summarizer._pending_task
    assert agent.event_manager.keys() == ["1", "2", "3", "4", "5"]
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == ["1..3", "4", "5"]
    assert agent.event_manager["1..3"].summary_text == "summary"
    summarizer._uninstall()


@pytest.mark.asyncio
async def test_fork_runs_middleware_without_recursive_forks_or_parent_statistics():
    agent, summarizer, ctx = setup()
    agent.runtime._last_prompt_tokens_actual = 42
    visits = []

    async def policy(request, nxt):
        visits.append(request)
        return await nxt(request)

    agent.event_manager.intercept("llm_call", policy)
    agent.llm.acall = AsyncMock(return_value=response())

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert summarizer._pending_task is not None
    await summarizer._pending_task
    assert len(visits) == 2
    assert agent.llm.acall.await_count == 1
    assert agent.runtime.last_prompt_tokens_actual == 42
    assert len(agent.event_manager.keys()) == 4
    summarizer._uninstall()


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["error", "tool", "empty", "stale"])
async def test_bad_or_stale_summary_never_collapses_history(problem, caplog):
    agent, summarizer, ctx = setup()
    if problem == "error":
        agent.llm.acall = AsyncMock(side_effect=RuntimeError("provider failed"))
    elif problem == "tool":
        agent.llm.acall = AsyncMock(
            return_value=response(
                "",
                tool_calls=[
                    ToolCall(
                        id="x", name="execute_python", arguments='{"code":"raise AssertionError"}'
                    )
                ],
            )
        )
    else:
        agent.llm.acall = AsyncMock(return_value=response("" if problem == "empty" else "summary"))

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert summarizer._pending_task is not None
    await summarizer._pending_task
    if problem == "stale":
        agent.event_manager.collapse("2", "3", "other summary")
    before = agent.event_manager.keys()
    summarizer._apply_pending_summary()
    assert agent.event_manager.keys() == before
    assert caplog.records
    summarizer._uninstall()


@pytest.mark.asyncio
async def test_uninstall_cancels_fork_and_removes_middleware():
    agent, summarizer, ctx = setup()
    entered = asyncio.Event()

    async def wait(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    agent.llm.acall = wait

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert summarizer._pending_task is not None
    task = summarizer._pending_task
    await asyncio.wait_for(entered.wait(), 1)
    summarizer._uninstall()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent.event_manager._middleware["llm_call"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("result,valid", [("summary", True), (42, False), ("", False)])
async def test_return_result_is_read_as_data_never_executed(result, valid):
    import json

    agent, summarizer, ctx = setup()
    agent.llm.acall = AsyncMock(
        return_value=response(
            "",
            tool_calls=[
                ToolCall(id="r", name="return_result", arguments=json.dumps({"result": result}))
            ],
        )
    )

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    await summarizer._pending_task
    assert summarizer._pending_summary == ("summary" if valid else None)
    summarizer._uninstall()


@pytest.mark.asyncio
@pytest.mark.parametrize("filtered,structured", [(True, False), (False, True)])
async def test_ineligible_request_uses_standalone_summary(filtered, structured, caplog):
    agent, summarizer, ctx = setup()
    ctx.filtered_history = filtered
    if structured:
        from pydantic import BaseModel

        ctx.params["output_model"] = BaseModel
    summarizer.summarize = AsyncMock(return_value="standalone summary")
    agent.llm.acall = AsyncMock()

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    await summarizer._pending_task
    summarizer.summarize.assert_awaited_once()
    agent.llm.acall.assert_not_awaited()
    assert "standalone" in caplog.text
    summarizer._uninstall()


@pytest.mark.asyncio
async def test_pending_summary_is_not_replaced_or_queued():
    agent, summarizer, ctx = setup()
    release = asyncio.Event()
    agent.llm.acall = AsyncMock(side_effect=lambda *a, **kw: None)

    async def summary(*args, **kwargs):
        await release.wait()
        return response()

    agent.llm.acall = summary

    async def core(request):
        request.response = response("parent")
        return request

    await agent.event_manager.run_middleware("llm_call", ctx, core)
    task = summarizer._pending_task
    await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert summarizer._pending_task is task
    release.set()
    await task
    await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert summarizer._pending_task is task
    summarizer._uninstall()


@pytest.mark.asyncio
async def test_install_does_not_change_parent_request_and_fork_uses_effective_client(monkeypatch):
    from nooa.runtime.actor import _current_llm_var, _current_method_var

    agent = Agent(llm=FakeLLMClient())
    effective = FakeLLMClient()
    for i in range(4):
        agent.event_manager.add(Message(content=f"fact {i}"))
    messages = [
        {"role": "system", "content": "stable"},
        CacheBoundary(),
        {"role": "user", "content": "live=1"},
    ]
    monkeypatch.setattr(agent.runtime, "_build_messages", AsyncMock(return_value=messages))
    calls = []

    async def call(messages, **kwargs):
        calls.append((list(messages), dict(kwargs)))
        return response("done")

    effective.acall = call

    async def work():
        pass

    llm_token = _current_llm_var.set(effective)
    method_token = _current_method_var.set(work)
    try:
        await agent.runtime.generate(tools=[], tool_choice="auto", temperature=0.3)
        summarizer = TokenBudgetSummarizer.install(
            agent, config=TokenBudgetConfig(max_tokens=100, preserve_recent=1)
        )
        policy_params = []

        async def policy(ctx, nxt):
            policy_params.append(dict(ctx.params))
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", policy)
        await agent.runtime.generate(tools=[], tool_choice="auto", temperature=0.3)
        assert summarizer._pending_task is not None
        await summarizer._pending_task
        assert calls[0] == calls[1]
        assert calls[2][1] == calls[1][1]
        assert calls[2][0][:-1] == calls[1][0]
        assert calls[2][0][1] is calls[1][0][1]
        assert "prompt_cache_key" not in policy_params[0]
        assert policy_params[1]["prompt_cache_key"] == calls[0][1]["prompt_cache_key"]
        summarizer._uninstall()
    finally:
        _current_method_var.reset(method_token)
        _current_llm_var.reset(llm_token)


@pytest.mark.asyncio
async def test_fork_preparation_failure_does_not_fail_parent(caplog):
    agent, summarizer, ctx = setup()

    class NotCopyable:
        def __deepcopy__(self, memo):
            raise TypeError("custom transport is not copyable")

    ctx.params["custom"] = NotCopyable()

    async def core(request):
        request.response = response("parent answer")
        return request

    result = await agent.event_manager.run_middleware("llm_call", ctx, core)
    assert result.response.content == "parent answer"
    assert summarizer._pending_task is None
    assert "fork" in caplog.text.lower()
    summarizer._uninstall()
