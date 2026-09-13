# Cache-sharing asynchronous summarization

## Question

Can a background summarizer reuse the parent's cached prompt instead of
rendering the same history as a new Markdown document under a different prompt?

## Plan for review

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
Estimated production delta: 150–200 lines, mostly the summarizer consumer.

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
results are logged and discarded, leaving history intact for a later attempt.

Total live spending across the protocol probe, initial failed attempt, and
successful installed probe: 12 calls, 95,472 input tokens and 786
output tokens. Provider-reported input includes cached tokens. These are
controlled synthetic probes, not general summary-quality evaluations.

## References

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Claude tools and caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
