# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delegated context must respect even very small caller-provided budgets."""

import json

import pytest
from nooa_cli.coding.context_rendering import render_delegated_context


@pytest.mark.parametrize("max_chars", [-10, 0, 1, 12, 13, 14, 50, 500])
def test_context_never_exceeds_character_budget(max_chars):
    """Truncation markers count toward the output budget, including at zero."""
    rendered = render_delegated_context({"payload": "x" * 200}, max_chars=max_chars)

    assert len(rendered) <= max(0, max_chars)
    if max_chars <= 0:
        assert rendered == ""
    if max_chars == 500:
        assert rendered == '{"payload": "' + "x" * 200 + '"}'


@pytest.mark.parametrize("size", [24, 25, 26])
@pytest.mark.parametrize("mapping", [False, True])
def test_item_limit_marks_only_actual_truncation(size, mapping):
    value = {str(i): i for i in range(size)} if mapping else list(range(size))
    rendered = render_delegated_context(value)
    assert ("items truncated" in rendered) == (size > 25)


def test_mapping_keys_remain_distinct_without_calling_arbitrary_repr():
    class Dangerous:
        def __repr__(self):
            raise AssertionError("must not render an arbitrary key")

    value = {1: "one", 2: "two", "<int: 1>": "literal", Dangerous(): "object"}
    rendered = json.loads(render_delegated_context(value))
    assert len(rendered) == 4
    assert set(rendered.values()) == {"one", "two", "literal", "object"}
    assert rendered["<int: 1>"] == "one"
    assert rendered["<int: 2>"] == "two"
