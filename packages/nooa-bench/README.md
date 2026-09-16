# nooa-bench

Coding benchmark agents and Harbor runner for
[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents), supporting SWE-bench
and Terminal-Bench tasks.

```bash
uv add nooa-bench
nemo-harbor --help
```

See the [main repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents) for
documentation.

Two agent variants are available through `nemo-harbor --agent-type`:

- `bench` — `BenchAgent` in `nooa_bench.bench_agent`: compact CodeAct baseline
  with automatic summarization and optional delegation.
- `rlm` — `RLMBenchAgent` in `nooa_bench.rlm_bench_agent`: the same capabilities
  with instructions emphasizing delegation for bounded work.

Both use `CodeActV2` with the single `python_cell` tool and return a structured `TaskResult`.
Its `how_to_verify` field describes concrete checks and expected results; commands
are optional. The `evidence` field records results the agent actually observed.
Both delegate through an awaited call returning a `TaskResult`; neither exposes
the interactive coding agent's background `spawn()` / job-handle API.
The strategy allows ten retries, uses a 1,800-second cell timeout, and has no
fixed iteration cap; configure the enclosing benchmark's time/token budget.
Workers use the same agent type, model client and working directory, with their
own execution context and shell. Delegation defaults to a maximum depth of four.
Passing a Todo gives the worker an independent task copy; successful worker
updates are merged after cleanup. Conflicts or worker-only dependencies raise
`DelegationMergeError`, retaining the completed `result` and full `worker_state`
for explicit reconciliation without rerunning the worker. Failed execution or
cleanup does not merge partial state. Task-local state stays on Todos, and automatic
summarization handles context maintenance. Method-writing tools are available in
both variants.

The runner writes `result.json`, `trajectory.json` and aggregate `behavior.json`
under `/logs/agent`, and the verification instructions to `/app/answer.txt`. Behavior
metrics count both Python tool names and exclude framework prefill. Set
`NOOA_INTERFACE_CHANGE_ID` to label a comparison; the default is `baseline`.
Set `NOOA_TASK_ID` to identify the task when logs share the `/logs/agent` path.
Exports include archived events after summarization, with the actual event IDs.
The original task inputs also remain in bounded, non-summarizable prompt context.
Metrics cover the controller's history; delegated workers keep separate histories
and their cells are not included. Recovery and retry metrics are omitted until
framework events carry explicit attempt linkage. Schema version 2 removes
unsupported error-code guesses from stdout. Code metrics count syntactic call
sites, not actual runtime loop iterations; fan-out recognizes direct and starred
`asyncio.gather` arguments, including comprehensions and same-cell list aliases.
Simple same-cell aliases of Todo, shell and repo objects are recognized; this is
not general cross-cell dataflow analysis. Reports with another schema, content
policy or unknown metrics are rejected; regenerate them from `trajectory.json`.
Delegation context redaction
uses credential-like mapping keys; arbitrary free text is not scrubbed.
Failure to generate the behavior report does not fail an otherwise completed
task. Agents close their shells; the runner closes the shared model client.

These are the current agent prompts and strategy. Reproducing a historical tech
report run requires its original code revision and configuration.

Apache-2.0 licensed.
