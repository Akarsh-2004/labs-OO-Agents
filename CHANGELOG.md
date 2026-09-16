# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Add `CodeActV2`, the single-`python_cell` strategy with in-cell `return_result`.
  The benchmark agents use it; `CodeActStrategy` remains the default.
  Its cacheable Python-cell context includes the execution namespace's typed
  stub without a second execution-context block.
- Trace explorer viewer requests now send configured viewer authentication and
  honor proxy environment settings, including `NO_PROXY` for direct access.
  Authenticated requests require HTTPS; cleartext URLs fail before sending a
  bearer token. Unauthenticated local HTTP access remains supported.
- `CurrentCall` is a mutable invocation record; strategies bind its event ID and
  live execution namespace with ordinary public-field assignment during setup.
- Breaking: remove `CodeActLiteStrategy` and its experimental exports. Use
  `CodeActStrategy` for the existing two-tool contract or `CodeActV2` for the
  single-tool contract. The evaluation CLI option is now `codeact_v2`.
- Preserve inline completion values in replay and archived events in benchmark
  trajectories; make Todo updates/restores atomic and delegation merge failures
  recoverable. Behavior reports use schema version 2; regenerate older reports
  from their trajectories before comparing results.
- Benchmark agents release resources through `aclose()` as well as `close()`;
  delegation prepares reference data before allocating a worker. Todo metadata
  and comment read-back methods are now included in model-facing documentation.
  Cancellation during shutdown is propagated only after background cleanup drains.
- `CodeActStrategy` remains the default strategy, but its model-facing behavior
  changes: revised delegation guidance, validated inline completion values in
  PythonOutput (None on validation failure), no replay of synthetic inline-return
  tool pairs, and explicit error/retry feedback for non-object tool arguments.
- Todo snapshot upgrades preserve legacy tasks, but downgrading to the previous
  implementation silently loses descriptions, active-task selection and comment
  IDs. Back up sessions before downgrading. `TodoVars` is now an alias for
  `PersistentVars`; helper-name keys must use explicit `get`/`set` access, and
  private/helper attribute writes are rejected. `InteractiveAgent.v` retains its
  separate `AgentVars` implementation.
- Delegation merge conflicts raise `DelegationMergeError` carrying the completed
  result and worker state. Benchmark agents no longer pre-seed a planning Todo;
  they expose tools through `python_cell_tools`, omit the `context_usage` block,
  and recreate the shell for each evaluation's working directory.

- Responses clients now honor the cached renderer's stable-prefix boundary by default,
  without a cache setting in the model registry. Requests without a usable boundary
  retain provider-default caching; `cache_breakpoint=None` opts out of NOOA markers.

- Security: the sandbox parent no longer unpickles worker bytes. Brokered `self.*`
  arguments, `self.x = value` assignments, cell return values and `return_result`
  payloads now cross as msgpack; rich values are rebuilt only from a fixed set of
  value types, numpy arrays, and the agent's declared pydantic models / dataclasses
  / enums (validated on the way in). Anything else is a `CellSerializationError`
  instead of code running in the parent. Adds the `msgpack` dependency.
- Breaking: custom CodeAct error formatters must implement
  `format(error, code=None, *, line_offset=0, max_error=None, tail_chars=None)`.
  Reduced legacy signatures are no longer supported.
- Breaking: sandboxed user-code failures are exposed as `SandboxExecutionError`;
  inspect `original_type`, `original_error`, and `diagnostic` for worker-side details.
- Add composable, context-scoped instrumentation hooks and trace-session scopes so hosts can observe NOOA execution without replacing native tracing.
- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
- Fixed: generator agent methods (`def`/`async def` containing `yield`) are now
  traced correctly. Their span previously covered only the *creation* of the
  generator, so LLM calls made by the body were recorded as children of whichever
  method drained it. Body calls now nest under the generator, and calls the
  consumer makes between yields do not.
- Breaking: a generator method with the `...` generation marker (including
  `yield ...`) now raises `TypeError` at class-creation time. Generation
  strategies commit one final result and do not define a stream protocol.
  Deterministic generators remain supported without `@strategy`.
