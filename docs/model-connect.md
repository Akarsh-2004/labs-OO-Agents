# Configure a model with NOOA Connect

`nooa connect` asks for approval of API-call costs upfront, checks the selected model with bounded
requests, then asks before saving its registry entry. The same `nooa.connect` library is available to
the TUI: frontends supply consent and display; the library supplies the plan,
UnifiedLLM checks and registry updates.

## CLI

Start the guided setup with no arguments:

```sh
uv run nooa connect
```

## Independent stages for agents

`--stage` runs without prompts and writes one JSON result to stdout. It never
saves implicitly. A generation stage runs directly within its configured limits;
there is no separate spending-approval dialogue. The interactive wizard remains
available when `--stage` is absent.

| Stage | Purpose |
|---|---|
| `discover` | List the endpoint's models (not proof of authentication). |
| `catalogue` | Read public model metadata; optional MODEL filters candidates. |
| `interfaces` | Check API formats and report which worked. |
| `plan` | Build an unsaved configuration and proposed requests; no network. |
| `routing` | Test one basic request with the chosen interface. |
| `tools` | Check whether the model produces the requested tool call; never execute it. |
| `reasoning` | Check declared levels supplied by `--levels-file`. |
| `session` | Check cache reuse and reasoning replay across turns. |
| `all` | Run the basic and conversation checks. |
| `save` | Write the entry from a JSON plan/result to a registry. |

For example:

```sh
uv run nooa connect --stage discover \
  --endpoint https://gateway.example/v1 --api-key-env MODEL_KEY

uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat --api-key-env MODEL_KEY

uv run nooa connect your-model --stage plan --as work \
  --endpoint https://gateway.example/v1 --api-style chat --api-key-env MODEL_KEY \
  > model-plan.json

uv run nooa connect --stage save --input model-plan.json --output llm_config.yaml
```

Use explicit endpoint/interface/key-variable options in stage mode; wizard
presets, pasted keys and `--no-probe` do not apply. `--budget-tokens` defaults to
65,536; `--output-tokens` controls basic check caps. Session checks retain their
separate documented cap. Redirect stdout to keep stage reports; `--output` is
the registry target for `save` only. Replacing an existing alias requires `--yes`
and prints a warning to stderr. Saving an untested plan does not validate it.

Reports have `version`, `stage`, `ok`, `data`, `checks`, `error`, and
`diagnostic_prompt`. Exit 0 means the stage met its criterion; 1 means failure
or missing evidence, not proof of unsupported features; 2 means invalid stage
options. Request acceptance alone is not success for tools or enabled reasoning
checks. A session requires both substantial cache reuse and retained reasoning.
The diagnostic prompt includes safe route/credential-variable names and outcomes,
not key values, raw provider error bodies, or returned reasoning. It asks an
agent to investigate, repair, and rerun the affected stage within configured
limits. Wizard failures print the same library-generated handoff; neither
frontend launches another agent automatically.

`data` contains the full library result, including the synthetic prompts in the
probe plan/provenance; `checks` omits those request bodies. Neither contains
returned reasoning or credential values. Local file failures retain their path;
YAML failures identify the file, line and column without echoing file contents.

## Library calls from NOOA agents

The CLI is a frontend to `nooa.connect`, not a subprocess requirement. Inside
an async NOOA agent method, use:

```python
from nooa import connect

proposal = connect.plan(
    "work", "your-model", "chat", "https://gateway.example/v1", "MODEL_KEY",
    budget_tokens=65536,
    reasoning_levels={"high": {"reasoning_effort": "high"}},
)
result = await connect.check_stage(proposal, "reasoning")
checks = result.entry["provenance"]["probes"]
handoff = connect.diagnostic_prompt("reasoning", result.entry, checks)
# Explicit persistence, when wanted:
# connect.write(result.entry, registry_path, alias=result.alias)
```

`check_stage` runs selected checks afresh and leaves the input plan unchanged.
Other public stages are `discover`, `catalogue`, `match_models`, `plan`,
`check_interfaces`, and `write`. Use `run_steps` for progress events on an approved
plan. The library never prints, prompts, reads stdin, or launches an agent;
frontends decide presentation and persistence.

## Interactive setup

Choose NVIDIA (build.nvidia.com), OpenAI, Anthropic, Google (Gemini), OpenRouter,
or a custom endpoint. Presets fill in the public server URL and key variable;
model names still come from the endpoint, not a bundled list. The order is
server, credentials, model selection, then automatic interface checks. It tries
Chat Completions, Responses and Anthropic Messages once each and offers only
interfaces that returned the expected response format. One success is selected
automatically; multiple successes give a choice. A timeout, authentication error
or rejected request is not labelled “unsupported.” If no check succeeds, the wizard
offers to change the key, edit the server and model, retry, or cancel without saving.
Failed attempts remain charged to the original approved budget; correcting the
connection does not increase it or ask for another spending approval. Exhausting
that budget ends setup without saving. Scripted `--yes` runs still fail without
prompting. A successful model listing is not described as authenticated: some
servers list their models publicly. Provider help banners are hidden during CLI
checks, while safe authentication, routing and timeout explanations remain visible.
A server listing models is not evidence that every model on it uses the same
interface. `--api-style` supplies an explicit choice and skips interface detection;
scripted `--yes --provider ...` setup uses the preset default if none is supplied.
The wizard asks what to call the model locally. It shows published model details
before asking to use them: context window, maximum reply length, reasoning levels,
and default reasoning level. Missing values say “Not listed.”
These come from OpenRouter's model listing; server limits may differ. The setup
check's output cap is shown separately and does not become the model's default.
Choose **use**, **edit**, **skip**, or **cancel**. Editing lets you change the
context window, maximum output, reasoning levels and default reasoning level;
Enter keeps each suggestion and `-` leaves a field unknown. The revised details
are shown again before you accept them, and saved edits are attributed to you.
Confirmed edits replace command-line limit/level settings; skipping the details
keeps any explicit command-line settings. Skip means continue without these
published details, not cancel. Cancel stops setup without saving.
It then shows the remaining checks and budget. The initial warning explains that setup
makes paid API calls and asks once for approval before any generation; there are no repeated approval prompts for these checks.
Each check shows progress and its result as it runs. Saving is a separate
confirmation; Ctrl-C cancels setup. A pasted key is used only for this session;
the saved entry names its environment variable, not its value.

If enabled reasoning-level checks succeed but return neither reasoning fields
nor reported reasoning tokens, Connect warns once before saving and lists the
affected levels. It suggests another API format or checking the server settings.
Disabled reasoning settings are exempt, as are rejected or skipped checks. A
server may hide reasoning information, so this warning does not claim that
reasoning is off. The saved results keep acceptance and reasoning observation
separate. The TUI can use `connect.unobserved_reasoning_levels(entry)` for the
same warning. This does not add calls or prove reasoning quality; the setup
question remains a small connection check.

Full setup also runs a three-call conversation check through the saved entry's
UnifiedLLM client and the default cached renderer. It seeds a longer reference
prompt, replays the actual reply, then repeats that saved history with a different
final question. It does not execute tools or invent reasoning state. It reports:

- Cache reads as a fraction of input tokens, whether explicit markers were sent,
  and whether the stable request prefix matched. Confirmation requires cache
  reads covering at least half of the seed's input; a small tool-schema-only hit
  is not enough.
- Whether reasoning appeared in replies, whether its readable/native state reached
  the follow-up's reasoning fields unchanged, and whether the selected reasoning
  controls survived on the wire. Ordinary text fallback does not count as retained
  reasoning. Missing evidence says “not confirmed,” not “reasoning is disabled.”

These calls use up to 2,048 output tokens each and reserve 43,008 estimated tokens
within the initial shared budget (65,536 by default). Basic checks retain their
200-token output cap. No retries run; checks stop rather than increase the approved
budget. These are estimates, not billing limits: servers can ignore caps. Use
`--budget-tokens` before setup to choose another budget. Small context windows,
incompatible reply caps, failures or insufficient budget leave the conversation
check unconfirmed or untested. Results include sanitized counts and flags only;
raw responses, signatures, encrypted state and captured bodies stay in memory.
Responses entries use `store: false` and request
`include: [reasoning.encrypted_content]` by default, with source `connect` recorded
in provenance. The wizard explains that encrypted reasoning carries reasoning
context between turns without requesting stored responses. This is a compatibility
request, not a guarantee of reasoning availability; current native OpenAI APIs may
return that state automatically. See [OpenAI's reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

A 400/422 explicitly rejecting `include` or `encrypted_content` disables the option
and records a sanitized rejection. Authentication, rate limits, server failures,
and invalid input-history errors do not change it. The saved `include: []` opt-out
omits the field on the wire, including on native endpoints. A routing rejection
can lead to one new capped check without the option, charged to the original
budget; no unchanged request is retried. A session rejection stops the conversation
check and leaves retention unconfirmed. Reconnecting to the same route preserves
the recorded rejection; changing routes tests the default again. Cache settings
are still tested without overrides.
`--no-probe` disables these calls too; `--yes` explicitly approves the selected
checks as well as saving. Library frontends opt in with `plan(..., session_checks=True)`.

If you do not want paid checks, use `--no-probe` or follow
[manual model configuration](model-configuration.md), optionally with the
`nooa-agent-authoring` skill. With `--no-probe`, interface selection is manual.

In a terminal, type part of a model name to filter a scrolling completion menu;
Tab selects a match. Large catalogues do not print every model into the prompt.
Provider, endpoint, API-format, alias, catalogue and confirmation prompts also
complete their available choices. Server URL suggestions include every valid
`api_base` in the existing target configuration file, as well as provider presets;
duplicates and URLs with embedded credentials are excluded. The URL menu opens
immediately: use arrow keys and Enter to pick a server, or type to filter or enter
a different URL. Key-variable completion uses environment
variable **names only**, never their values. Secret entry is masked and has no
completion or history. Defaults appear as faint suggestions: Enter accepts one,
typing replaces it, and Right Arrow brings it into the editor. Arrow keys,
Home/End, Backspace and Delete edit the current answer. Existing alias names in
the target file show a warning while typing; replacement still requires confirmation.
The terminal groups setup into four steps, with a provider menu and F1 help.
Piped input uses plain line prompts.

Flags can prefill answers or support scripted setup:

```sh
uv run nooa connect --provider nvidia

uv run nooa connect gateway/model --as work-model \
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
In the wizard, enter `-` at the key-variable prompt for no authentication.

Preset endpoints follow the public connection guides for
[NVIDIA](https://docs.api.nvidia.com/nim/docs/api-quickstart),
[OpenAI](https://developers.openai.com/api/reference/overview),
[Anthropic](https://platform.claude.com/docs/en/api/overview),
[Google's OpenAI-compatible API](https://ai.google.dev/gemini-api/docs/openai), and
[OpenRouter](https://openrouter.ai/docs/quickstart). They do not certify that a
particular model supports every optional feature; the approved probes check that
endpoint. Frontends can reuse the defaults through `connect.PROVIDERS`.

OpenRouter metadata supplies candidate model names, context and output limits,
prices and reasoning levels where present. Confirm the candidate; a matching
name does not prove that a gateway exposes the same capabilities. Ambiguous
matches require a choice, including with `--yes`. Use `--catalogue-model` for an
explicit choice or `--no-catalogue` to leave it unknown.

Use `--no-probe` to save without model calls, or `--probe minimal` for routing
only (up to three interface attempts unless `--api-style` is supplied).
The default plan also checks tools and each proposed reasoning level. The
selected interface's successful routing request is reused, not sent twice.
Basic checks use 200 output tokens per call, no retries, and a 30-second total
deadline per probe (120 seconds for each conversation-check call). The CLI uses
the fixed budget approved at the beginning. Each basic request reserves its
output cap plus 512 estimated input tokens. An explicit `--budget-tokens` limits
the entire setup, including interface detection; it is never increased. Connect
warns before the remaining checks if that limit is too small. Larger reported
usage increases the charge. The next request is skipped if the budget would be
exceeded, and a warning lists checks left undone before saving. These are
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
registry's reasoning-level mechanism. Every generation check uses UnifiedLLM,
constructed from the unsaved entry through the same factory as `get_llm_client`.
That tests runtime routing, settings translation and response parsing, not just
whether the server accepts a hand-built request. Context and maximum-output limits are
metadata, not permission to generate that many tokens on each call.

## Files and reconnecting

The default is `llm_config.yaml` in NOOA's user configuration directory. Connect
warns before overwriting an existing alias, including a hand-written one, and
asks for confirmation before replacing it. Interface checks may already have
run before the local alias is chosen. `--yes` skips save/overwrite confirmation;
it still prints the overwrite warning. Other aliases and surrounding
comments stay intact. Writes replace the file atomically.

Unchanged accepted UnifiedLLM probes are reused when reconnecting. Older direct-HTTP
checks are repeated: they did not test the runtime. Changing the route or
level declarations changes which probe requests can be reused; unchanged requests
on the same route remain reusable. `--output` chooses another
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
# Frontends own the cost warning or approval. plan() above performs no I/O;
# run() below sends the approved requests and may incur charges.
result = await connect.run(proposal, approved="minimal", api_key=temporary_key)
# Separately obtain approval to save.
connect.write(result.entry, destination, alias=result.alias)
```

`run` accepts `all`, `minimal` or `none`. Cancelling its task cancels the HTTP call.
For inline feedback, consume `run_steps()` instead: it yields `ProbeUpdate`
objects before and after each check, followed by the final `ConnectResult`.
Use `contextlib.aclosing` if the frontend may stop reading early. `run()` uses
the same iterator internally, so CLI and TUI checks cannot diverge.
For interface detection, consume `check_interfaces(alias, model, api_base,
api_key_env, ...)`. It yields `ProbeUpdate` events named for each interface and
then an `InterfaceResult` containing `accepted`, the per-interface `results`,
and `tokens_charged_to_budget`. Offer only `accepted`, pass the chosen result's
entry to `plan(existing_entry=...)`, and deduct that charge from the remaining
budget before running additional probes. This reuses the exact accepted routing
request; level and tool requests still need their own checks.
There are no callbacks, terminal imports, prompts, agent instances or tool
execution. Discovery uses HTTPX; generation checks lazily load UnifiedLLM and
the current runtime (LiteLLM by default). No temporary registry entries or global
registry changes are needed. Each checked client is closed even if its call fails.
The TUI keeps model selection, confirmation, secret persistence and switching;
it can call these async functions directly without invoking Click or a subprocess.
Its existing Ollama-specific adapter remains separate from these three API styles.

## What the observations mean

Each record distinguishes successful runtime calls and observed reasoning.
The stored request is the planned input, not a claim that every field survived
runtime translation unchanged. Reasoning observations come from readable response
text or reported reasoning-token usage. A successful call alone does not establish
that a setting had an effect.
HTTP 400 is recorded as rejected, not unsupported. Auth, timeout and transient
failures remain untested; failed routing or auth stops subsequent calls within
that interface's plan. Interface detection still tries the other styles within
the same budget, because their authentication conventions differ. Reasoning
not being visible can be normal, especially at a disabled level. Returned tool
calls are inspected as data and never executed. Raw model responses, server error
bodies and credential headers are not retained. Limits and defaults keep their
catalogue source and are explicitly marked as not probed.
An endpoint speaking Responses does not imply it accepts explicit cache fields;
Connect leaves OpenAI explicit caching unset until that is established separately.
Responses checks use `store: false`; any other request defaults come from the
same client an agent will use. Connect does not add optional encrypted-reasoning
settings to the saved entry.

## Code walkthrough: what and why

- `src/nooa/connect.py`: plans data first so either frontend can obtain consent;
  runs bounded UnifiedLLM calls through the registry's shared client factory;
  updates one alias while retaining other entries and comments.
- `packages/nooa-cli/src/nooa_cli/commands/connect.py`: argument parsing, choices,
  preview and approval only. The TUI does not need to invoke this command.
- `tests/test_connect.py` and CLI tests: exercise approval, budgets, exact HTTP
  bodies, key privacy, cancellation, targeted writes and the real registry on main.

Request-shape references: [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [OpenRouter model metadata](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties).
