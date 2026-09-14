# Cache-sharing background summaries

`TokenBudgetSummarizer` makes a background request using the parent's messages,
effective client, tools, settings and cache key, followed by a summary instruction.
It does not clone the agent or execute tools. This is its only automatic path;
there is no mode setting, separate summarizer client or standalone fallback.

The collapse range is chosen before the parent request. Recent events and the
response just produced remain active. The parent continues while the summary
runs. At most one summary waits or runs at a time. A completed summary is applied
before the next turn only if the original event IDs still match.

The summary can be plain text or one `return_result` call containing a nonempty
string. That call is read as data, never executed. Empty, malformed, truncated
or executable-tool replies leave the history unchanged. After two consecutive
failed forks the hook is removed, with a warning, to stop repeated paid failures.
Fix the reported cause before reinstalling the summarizer.

Filtered-history requests are skipped: the fork cannot summarize events it did
not see. A known filter produces a warning at installation; later scoped filters
warn on first use, once per summarizer. The runtime's context overflow safety net
still applies. Structured-output parents remove `output_model` from the fork so
it can return text; changing that schema may reduce cache reuse. Tools stay the
same, including typed `return_result` schemas; an incompatible reply is rejected.

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config.summarizer_config import TokenBudgetConfig

summarizer = TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
# Before closing the shared client:
await summarizer.aclose()
```

`reuse_parent_prefix` and a separate `llm=` are no longer supported. The separate
`MethodSummarizer` feature still summarizes completed methods through its own
rendered input; it is not used as a token-budget fallback.

## Code walkthrough: what and why

- `src/nooa/agents/summarization.py`: intercepts the completed request so the
  fork uses the actual parent prefix, without re-rendering it. Copies only dict
  and list containers, sharing tools and immutable response/boundary objects.
  Checks failure, cancellation and event identities before collapsing history.
- `src/nooa/runtime/actor.py` and `middleware.py`: expose the effective client,
  cache key and filtered-history status at the existing middleware boundary.
  The fork uses the same middleware chain, with a task-local recursion guard.
- `tests/agents/test_forked_summarizer*.py`: check parent-request parity, HTTP
  prefix equality, background execution, ownership, safe output handling and
  collapse timing. These are permanent offline tests, not an experiment.

Offline tests verify that the fork preserves the parent request's prefix.
They do not measure provider cache hits or summary quality. Cache lifetime,
routing, changed suffixes and structured-output settings can lower reuse.
Keep deployment-specific measurements with the configuration used to run them.
