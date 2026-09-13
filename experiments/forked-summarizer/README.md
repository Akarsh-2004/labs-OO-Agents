# Cache-sharing asynchronous summarization

## Question

Can a background summarizer reuse the parent's cached prompt instead of
rendering the same history as a new Markdown document under a different prompt?

## Design reviewed with Wren

Use the existing `llm_call` middleware boundary for token-budget summarization.
After a successful parent request, launch one background call with that request's
message list, client, tools, cache key, and sampling settings. Append a summary
instruction after its cache boundary. Do not change effort, tool choice, or schemas.
The fork is a conversation branch, not a copy of the Python agent or its tools.

Select the collapse range before the parent call. It excludes the response and
tool work just produced, so summarization lags one turn. Never replace events
that arrived while the summary was running. Apply a completed summary only at
the existing BeforeTurn boundary. Failures must leave the history untouched.

The fork runs through the parent's middleware chain with a task-local recursion
guard; it must never execute tools. Preserve response objects and the boundary,
copy mutable message containers at fork time, and do not rebuild parent context
or update parent token statistics. Retain the existing standalone summarizer
for explicitly different clients, structured-output requests, or filtered history.
Method-completion summarization remains standalone in this first change.

Runtime changes should be limited to exposing the effective client and whether
history is filtered on `LLMCallContext`, and making the effective cache key
available there after dispatch. No new UnifiedLLM abstraction or provider rules.
The implementation is concentrated in the summarizer consumer. The interactive
coding agent also cancels pending summaries before closing its shared client,
and model-limit updates preserve the selected summarization mode.

## Usage

```python
summarizer = TokenBudgetSummarizer.install(
    agent, config=TokenBudgetConfig(max_tokens=80_000, preserve_recent=10)
)
# Default: reuse the parent's prefix. For independent summarization instead:
# TokenBudgetConfig(..., reuse_parent_prefix=False)
# On shutdown, before closing agent.llm:
await summarizer.aclose()
```

Same-model token-budget summaries benefit most when replacing most of a long
history. Cache-read tokens still cost money: a cached whole-history fork is not
necessarily cheaper than a small standalone range. Custom middleware is rerun
and may edit the request; changing the prefix can reduce cache reuse. Context
queries use the standalone path so unseen events are not summarized from a
partial parent request. Normal rendering/truncation limits still apply.

## Experiment

First test the protocol using real renderer output and unchanged request settings:
one parent request, a fork with a short summary suffix, and a standalone summary
with a different prefix. Include a trailing dynamic context block. Run OpenAI
Responses and Claude Sonnet through NVIDIA Inference Hub, once each. No retries;
bounded output. Do not print credentials, history, or opaque state.

Measure cached input / total input and output length. This tests cache reuse,
not semantic summary quality on arbitrary histories. A 99% cache hit is not a
guarantee: the summary instruction, live suffix, new history, expiration and cache
routing all matter. Even 99% cached input still costs tokens and output work.

## Tests before implementation

- Wire prefixes, tools, effort and cache key match; dynamic context stays trailing.
- Parent proceeds while the summary is blocked; no recursive forks or tool execution.
- Recent/new events survive; stale overlapping summaries are discarded.
- Native response objects retain identity; mutable dictionaries are detached.
- Failures, cancellation, filtered history and explicit alternative models are clear.

## Run

`uv run --env-file /path/to/credentials.env python experiments/forked-summarizer/probe.py`

Requires `NVIDIA_INFERENCE_API_KEY`. The opt-in script spends inference tokens.

## Results

NVIDIA Inference Hub, 2026-09-13, one run per variant. Both probes include a
trailing dynamic context block. No retries or artificial cache-read assertions.

| Probe | Provider | Cached / input tokens | Cache fraction |
|---|---|---:|---:|
| Protocol fork | OpenAI | 5,772 / 5,831 | 99.0% |
| Protocol standalone | OpenAI | 0 / 5,788 | 0% |
| Protocol fork | Claude Sonnet | 10,106 / 10,174 | 99.3% |
| Protocol standalone | Claude Sonnet | 0 / 10,124 | 0% |
| Installed CodeAct summarizer | OpenAI | 7,141 / 7,214 | 99.0% |
| Installed CodeAct summarizer | Claude Sonnet | 12,213 / 12,399 | 98.5% |

The installed test asserts that the summary preserves Tuesday, 42 units, and
Alex, then applies it through the existing safe-boundary method. See
`runtime_probe.py`. It uses untruncated synthetic notes; larger histories should
reduce the fraction spent on the short summary instruction, but cache placement,
expiry and routing still determine the actual hit rate.

An initial installed OpenAI probe found that CodeAct returns the summary in a
`return_result` call even when the suffix asks for prose. The safe rejection
worked but produced no summary. After a failing regression test, the consumer
now accepts exactly one `return_result` containing a nonempty string as data;
it does not execute any tool. Executable/mixed calls and malformed or empty
results now trigger one standalone attempt with a warning about losing cache
reuse. Provider or middleware exceptions do not trigger that fallback. If both
summary attempts fail, history stays intact. The successful live measurements
above predate these fallback/ownership/shutdown review fixes; the SDK request
shape is unchanged and checked offline, not presented as another paid rerun.

Wren's implementation review found that deep-copying a bound Tool could clone
its owner. A small helper now copies only dict/list containers, sharing tools,
clients and read-only response/boundary objects. Tests verify zero owner copies,
unchanged parent request settings, detached nested dictionaries, async progress,
one pending task, no recursive forks, safe result decoding, stale-range handling,
fallback, and cancellation. The first full offline run passed 7,694 tests;
final review-fix validation is recorded after the rerun.

Total live spending across the protocol probe, initial failed attempt, and
successful installed probe: 12 calls, 95,472 input tokens and 786
output tokens. Provider-reported input includes cached tokens. These are
controlled synthetic probes, not general summary-quality evaluations.

## References

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Claude tools and caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
