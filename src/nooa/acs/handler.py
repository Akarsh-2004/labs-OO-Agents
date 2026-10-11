# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any

from nooa.acs.schema import ACSDecision, ACSHook

logger = logging.getLogger(__name__)

class ACSHandler(ABC):
    """
    Base interface for an ACS Guardian.
    """
    
    @abstractmethod
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        """
        Evaluate a hook and return a governance decision.
        """
        pass


class DefaultLocalHandler(ACSHandler):
    """
    NOOA-as-Guardian: Always allows execution. 
    This represents the default unrestricted local development mode.
    """
    
    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        return ACSDecision(status="allow")


class AuditOnlyHandler(ACSHandler):
    """
    Audit-only Guardian: Records hooks to a provided sink function,
    but always allows execution without blocking.
    """
    
    def __init__(self, sink_func: Callable[[ACSHook], Any]):
        self.sink_func = sink_func
        self._delivery_tasks: set[asyncio.Task[None]] = set()

    async def evaluate(self, hook: ACSHook) -> ACSDecision:
        async def deliver() -> None:
            try:
                if asyncio.iscoroutinefunction(self.sink_func):
                    result = self.sink_func(hook)
                else:
                    result = await asyncio.to_thread(self.sink_func, hook)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("Audit sink delivery failed")

        task = asyncio.create_task(deliver())
        self._delivery_tasks.add(task)
        task.add_done_callback(self._delivery_tasks.discard)
        return ACSDecision(status="allow")
