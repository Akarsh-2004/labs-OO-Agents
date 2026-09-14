# Configure a model with NOOA Connect

`nooa connect` prepares a registry entry, shows it for approval, optionally tests
small requests, then saves it. The same `nooa.connect` library is available to
the TUI: frontends supply consent and display; the library supplies the plan,
HTTP probes and registry updates.

## CLI

```sh
nooa connect gateway/model --as work-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY
```

The model argument is the exact ID used by the endpoint, without the additional
LiteLLM routing prefix. API styles are `chat`, `responses`, and `anthropic`.
Omit MODEL to list and select from the endpoint's models. Discovery tries `/models`
and then `/v1/models` if a root endpoint returns 404. For Anthropic, explicitly
name the appropriate key environment variable. `--prompt-key` reads a masked key
for this setup only; Connect never saves it or changes environment variables.
An empty `--api-key-env ''` supports local servers without authentication.

OpenRouter metadata supplies candidate model names, context and output limits,
prices and reasoning levels where present. Confirm the candidate; a matching
name does not prove that a gateway exposes the same capabilities. Ambiguous
matches require a choice, including with `--yes`. Use `--catalogue-model` for an
explicit choice or `--no-catalogue` to leave it unknown.

Use `--no-probe` to save without model calls, or `--probe minimal` for one routing
call. The default plan includes routing, tools, and each proposed reasoning
level. It uses 200 output tokens per call, no retries, and a 30-second total
deadline per probe. `--budget-tokens` defaults to 4096. Each request reserves its
output cap plus 512 estimated input tokens; larger reported usage increases the
charge. The next request is skipped if the budget would be exceeded. These are
estimates, not billing caps: a gateway may ignore an output limit. There are no
context-window or maximum-output capacity probes.

When metadata has no levels, they remain unknown. You can supply candidates:

```sh
nooa connect gateway/model --as work-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-catalogue \
  --reasoning-template effort --levels low,medium,high
```

Other templates are `adaptive`, `budget`, `toggle`, and `thinking`. They are
candidate request shapes, not provider support tables. For exact route settings,
`--levels-file` accepts a YAML mapping of labels to whole request blocks instead.
Quote labels such as `"off"` that YAML otherwise treats as booleans. A template
whose output allocation exceeds the approved cap is left untested; Connect does
not silently increase the budget. `--context-window` supplies a known limit with
user provenance; an unknown limit remains absent and the current runtime's
fallback applies.

The saved entry uses the current `model_name`/`client_type` schema and the
registry's reasoning-level mechanism. HTTP probes test the endpoint directly,
not LiteLLM's translation, so acceptance does not prove that the runtime will
forward an unfamiliar setting unchanged. Context and maximum-output limits are
metadata, not permission to generate that many tokens on each call.

## Files and reconnecting

The default is `llm_config.yaml` in NOOA's user configuration directory. Connect
warns before overwriting an existing alias, including a hand-written one, and
asks for confirmation before any paid probes. `--yes` approves both testing and
saving; it still prints the overwrite warning. Other aliases and surrounding
comments stay intact. Writes replace the file atomically.

Unchanged accepted probes are reused when reconnecting. Changing the route or
level declarations discards previous probe results. `--output` chooses another
file; load custom paths with `NEMO_OO_LLM_CONFIG` or `reload_registry(path)`.
Later configuration layers can override the alias; inspect `llm_config_chain()`
if a saved entry does not take effect.

## Library interface for the TUI

```python
from nooa import connect

# Optional, before selecting the model (no generation calls):
discovery = await connect.discover("https://gateway.example", api_key=temporary_key)
# The frontend selects an ID from discovery.models.
proposal = connect.plan(
    "work-model", selected_model, "chat", discovery.api_base, "MY_MODEL_KEY",
    catalogue=None,
)
# Show proposal.entry, proposal.probes and its estimates in the frontend.
# Obtain user approval there; this call makes no requests and reads no keys.
result = await connect.run(proposal, approved="minimal", api_key=temporary_key)
# Separately obtain approval to save.
connect.write(result.entry, destination, alias=result.alias)
```

`run` accepts `all`, `minimal` or `none`. Cancelling its task cancels the HTTP call.
There are no callbacks, terminal imports, prompts, agent instances or tool
execution. Probes use HTTPX directly, not an SDK or LiteLLM. Importing the package
still triggers NOOA's existing eager imports; changing that is separate work.
The TUI keeps model selection, confirmation, secret persistence and switching;
it can call these async functions directly without invoking Click or a subprocess.
Its existing Ollama-specific adapter remains separate from these three API styles.

## What the observations mean

Each record distinguishes fields sent, HTTP acceptance and observed reasoning.
HTTP 400 is recorded as rejected, not unsupported. Auth, timeout and transient
failures remain untested; failed routing or auth stops subsequent calls. Reasoning
not being visible can be normal, especially at a disabled level. Returned tool
calls are inspected as data and never executed. Raw model responses, server error
bodies and credential headers are not retained. Limits and defaults keep their
catalogue source and are explicitly marked as not probed.

## Code walkthrough: what and why

- `src/nooa/connect.py`: plans data first so either frontend can obtain consent;
  sends bounded HTTP requests so no new SDK or runtime dependency is required;
  updates one alias while retaining other entries and comments.
- `packages/nooa-cli/src/nooa_cli/commands/connect.py`: argument parsing, choices,
  preview and approval only. The TUI does not need to invoke this command.
- `tests/test_connect.py` and CLI tests: exercise approval, budgets, exact HTTP
  bodies, key privacy, cancellation, targeted writes and the real registry on main.

Request-shape references: [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [OpenRouter model metadata](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties).
