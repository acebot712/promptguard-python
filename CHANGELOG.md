# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`Unreleased` holds work that is merged but not yet published. Move entries into
a dated version section when a release goes out — an `Unreleased` block that
survives three releases is a changelog nobody is maintaining.


## [Unreleased]

### Fixed

- **`PromptGuardCallbackHandler` (LangChain) failed every call it was attached
  to.** langchain-core reads `ignore_chat_model`, `ignore_chain`, ... and
  `run_inline` off a handler before each event, and the duck-typed handler
  defined none of them, so the first callback raised `AttributeError` and
  `raise_error = True` turned that into a failed LLM call. The handler now
  subclasses langchain-core's `BaseCallbackHandler` whenever LangChain is
  installed, which also covers events it does not implement (streaming's
  `on_llm_new_token` failed the same way). Requires no change to user code.
- **The handler crashed on chains and LangGraph graphs.** Runnables pass
  `serialized=None` to `on_chain_start`; every `*_start` callback now accepts
  it, and chain context falls back to the run's `name`. LangGraph works:
  blocks raise `PromptGuardBlockedError` out of `graph.invoke` / `ainvoke`, and
  tool inputs and outputs are scanned. On Python 3.10 an async graph reaches
  the handler only if each node forwards its `config` to the model call.
- **The CrewAI example registered nothing.** It passed `before_kickoff=` /
  `after_kickoff=`, which `Crew` silently ignores; the fields are the lists
  `before_kickoff_callbacks` / `after_kickoff_callbacks` (module docstring and
  README).

### Changed

- The `dev` extra now installs `langchain-core` and `langgraph`, and
  `tests/test_langchain_core.py` drives the handler through the real framework.
  The existing integration tests call the handler's methods directly, which is
  why none of the above was caught.

- **Framework integrations send the framework's real conversation id as
  `session_id`, or none** (acebot712/promptguard#956). The LangChain handler
  used to send each call's run id, and the LlamaIndex handler each event's id,
  so no two scans ever shared a session. The LangChain handler now sends the
  call's `thread_id` (from LangGraph, or from `config["metadata"]`, where
  `session_id` and `conversation_id` are read too) on the start scan and the
  end scan alike; `PromptGuardRunnable` reads the same from its invoke config.
  LlamaIndex events name no conversation, so its handler and
  `PromptGuardQueryGuard` send one only when given `session_id=`. A call with
  no conversation now sends no `session_id` at all. See "Conversations" in the
  README for each integration.

- `SECURITY.md` links to the published disclosure policy
  (https://promptguard.co/security#reporting-a-vulnerability) instead of
  keeping its own copy of the response times.

### Added

- `session_id=` on `PromptGuardCallbackHandler` (LangChain and LlamaIndex),
  `PromptGuardRunnable`, `PromptGuardQueryGuard` and CrewAI's
  `PromptGuardGuardrail`, for a conversation the framework does not name
  (acebot712/promptguard#956); the call's own id wins over it.
- `session_id=` (a function of the ASGI scope) and `session_id_header=` on
  the FastAPI / ASGI middleware, to send the request's conversation as
  `session_id`. Read it from the authenticated session, or from a header only
  your own auth layer or a trusted proxy sets, never from browser input.
- A session id over 255 characters is dropped with a debug log and the scan
  goes ahead without one; passed to a constructor, it is a `ValueError`.
- Auto-instrumentation sends an OpenAI Responses call's `conversation` as
  `session_id`.

- **FastAPI / ASGI middleware**, `promptguard.integrations.fastapi.PromptGuardMiddleware`
  (acebot712/promptguard#945). It scans the fields in `DEFAULT_PROMPT_FIELDS`
  (configurable) of any request body on the selected paths with the async Guard
  client, and answers a block with `403` before the route runs; otherwise the
  body is replayed to the app unchanged, chunked and compressed bodies
  included. It reads JSON (top-level arrays too), `text/plain` and form bodies
  in their declared charset, and decompresses gzip/deflate; multipart is not
  scanned. Content blocks and tool calls go to the Guard API as they are.
  `paths` and `exclude_paths` match whole segments of the route path, with
  `root_path` stripped, so mounted apps work. A body over `max_body_bytes`
  (`DEFAULT_MAX_BODY_BYTES`, raw or decompressed) gets `413`. A Guard
  `400`/`413`/`422` is refused in every mode; a transport error, timeout, `5xx`
  or `429` fails open with a `WARNING` carrying
  `promptguard_event="guard_unavailable_fail_open"` (`fail_open=False` answers
  `503`), and a `401`/`403` does the same after one `ERROR`. A redact decision
  is refused in enforce mode. `anyio`, already installed by `httpx`, is now
  declared, because the middleware parses large bodies in a worker thread
  with it.
- **`instrumentation_report()["middleware"]`**: the request-edge middleware
  wrapping an app in the process (`["fastapi"]` once the app has started).
  This is cross-SDK contract v1.7.0, which adds the key to the report;
  `verify()` counts a live middleware as coverage.

## [2.3.0] — 2026-10-10

### Added

Generated API types re-synced from the platform's OpenAPI spec (#41). Types
only, additive only: nothing was removed or renamed, and no runtime behaviour
changed. All of it lives in `promptguard.generated.api_types`.

- **`GuardResponse.unavailable`**, a list of the new `UnavailableCheck`
  (`detector`, `reason`): detectors the scan would have run but the deployment
  cannot, because their backing service is not configured. An `allow` with a
  non-empty `unavailable` does not mean every check passed.
- **`GuardRequest.device_findings`**, a list of the new `DeviceFinding`
  (`category`, `count`): what the calling Device's own masker caught before
  sending. Optional; it does not change the decision.
- **`ThreatType`**, the literal of threat categories the API returns.
- Device-facing response types: `DeviceIdentity`, `DeviceView`,
  `DeviceViewList`, `DeviceRevokedResponse`, `DeviceShadowException`,
  `DeviceShadowExceptionList`, `DeviceToolRequest`, `DeviceToolRequestList`,
  `ActiveGrant` and `ActiveGrantList`.
- Proxy response types: `ProxyChatCompletion`, `ProxyResponsesObject`,
  `ProxyModelList`, `ProxyAnthropicMessage` and `ProxyTokenCount`.
- Other response types: `UsageStatsResponse`, `ApiKeyToggledResponse`,
  `AgentSecurityHealthResponse`, `AgentSessionEndedResponse`,
  `ApiHealthResponse`, `ApiRootResponse`, `ApiVersionProbeResponse` and
  `WebhookAckResponse`.

### Changed

- Development only: ruff 0.16.9, CI reads its ruff pin from `pyproject.toml`,
  and the type sync formats what it generates (#42, #43, #44). Nothing in the
  installed package is affected.

## [2.2.1] — 2026-09-18

### Changed

- **Generated types and the cross-SDK contract re-synced from the platform API**
  (#40). Types only: no runtime behaviour changes. Published so the package on
  PyPI matches the contract the API now serves.


## [2.2.0] — 2026-09-06

### Added

- **`tokenize` is a valid `pii_detection.mode` in the generated types.** The
  `PiiDetectionConfig` mode literal listed `redact`, `mask` and `block` only, so
  a type-checked project passing `tokenize` — the reversible mode the API has
  always accepted — failed at the type checker rather than at the API. The same
  gap was fixed in the CLI's `policy apply` in cli-v2.1.0.
- **`multi_turn_drift`** appears on the detector-configuration type.

### Changed

- The managed-update-policy docstring names the `shadow_fleet_management`
  entitlement, which is what the API actually reads; it previously said
  `shadow_ai_fleet`.

### Removed

- **`ApiKeyFullResponse` is gone from the generated types**, following the
  removal of the key-reveal endpoint it described. Not a major bump: the
  generated module declares itself type-only, is not re-exported from
  `promptguard`, and the name is not in the package's public surface. It was
  still reachable as `promptguard.generated.api_types.ApiKeyFullResponse`, so a
  project that imported it directly will need to drop it.

## [2.1.0] — 2026-08-30

### Added

- Generated API types for the full per-guardrail project configuration:
  `GuardrailsConfig`, `LevelConfig`, `PIIDetectionConfig`, `ToxicityConfig` and
  `ToggleOnlyConfig`. Synced from the platform's OpenAPI spec, so a client
  configuring guardrails now has types for the shape the API accepts instead of
  hand-writing dicts.

Additive only — nothing in 2.0.0 changed behaviour or signature.

## [2.0.0] — 2026-08-24

### Fixed

- **The red team client was unreachable for every customer.** Every method on
  `client.redteam` targeted `/internal/redteam`, which is the platform-admin
  plane and rejects an API key outright — so the namespace could not have
  worked for anyone holding a `pg_live_` key, whatever they did. It now targets
  the customer-facing `/api/v1/security-testing`.

### Removed

- **BREAKING — three namespaces that could only ever 404 are gone**, in both
  the sync and async clients: `completions`, `embeddings` and `scrape`, along
  with `redteam.run_autonomous()` and `redteam.intelligence_stats()`. Nothing
  that worked has been taken away — there was never a working implementation
  behind them — but removing public attributes from a published client is a
  backwards-incompatible change, so it takes a major version.

  There is no replacement, because there was nothing to replace. For chat
  traffic, use `client.chat.completions.create(...)`, which is unchanged.


## [1.13.0] — 2026-08-16

### Added

- **`promptguard.verify()` — a positive check that protection is actually
  live.** `init()` returning cleanly has never meant anything is being scanned:
  the SDK fails open, so a rejected API key, an unreachable Guard API or a
  provider SDK we never hooked all leave an application that runs perfectly and
  blocks nothing. Each of those already logged a warning, but a warning in a log
  nobody tails is indistinguishable from silence. `verify()` makes the real
  calls — reachability, authentication, a live injection probe, a PII probe —
  and returns what came back, so a deployment can assert it instead of assuming
  it: `assert promptguard.verify()["ok"]`. The checks and their names mirror
  `promptguard verify` in the CLI so the two agree on what "working" means. It
  never raises for a failed check, so a CI step sees every problem at once
  rather than only the first.

## [1.12.0] — 2026-08-11

### Fixed

- **Gemini calls made through Google's current SDK were not being scanned.**
  Auto-instrumentation patched `google-generativeai`, which Google deprecated
  when Gemini 2.0 shipped. The current package is `google-genai`, a different
  import path, so `promptguard.init()` returned no error and protected nothing.
  Both SDKs are now patched; the deprecated one is kept because customers are
  still on it. The new patch additionally scans `config.system_instruction` and
  can redact (the legacy call signature could not).

### Added

- **An installed provider SDK we did not hook is now named at startup.**
  `init()` previously warned only when it patched *nothing*, so a customer with
  `openai` and `google-genai` installed saw a healthy startup and no mention of
  their unscanned Gemini traffic. Each patch module declares what it detects,
  and anything installed-but-unhooked is warned about by name with the fix.
- `instrumentation_report()` — the same facts as data, so coverage can be
  asserted in your own CI rather than read from startup logs:
  `assert promptguard.instrumentation_report()["detected_unpatched"] == []`.


<!-- Entries below are reconstructed from git tags: this file did not exist
     until 2026-08-11, so the tag subject is all the detail anyone recorded.
     Accurate about WHAT shipped, thin on WHY. Entries from here on are
     written when the release is cut -- scripts/check_changelog.py fails the
     release if the version being published has no section. -->

## [1.11.1] — 2026-08-03

### Changed

- v1.11.1 — dependency updates

## [1.11.0] — 2026-07-13

### Changed

- chore(release): bump version to 1.11.0 (LangChain + LlamaIndex adapters) (#19)

## [1.10.0] — 2026-07-12

### Changed

- chore(release): 1.10.0

## [1.9.0] — 2026-06-01

### Changed

- release: v1.9.0 — proxy-suffix auto-fix, retry controls, SecurityScanResult (#14)

## [1.8.1] — 2026-04-19

### Changed

- chore(release): 1.8.1 (#7)

## [1.8.0] — 2026-04-11

### Changed

- chore: bump version to 1.8.0

## [1.7.1] — 2026-04-10

### Changed

- chore: update model references from gpt-4o to gpt-5-nano and bump to v1.7.1

## [1.7.0] — 2026-04-07

### Changed

- feat: add API contract testing, OpenAPI validation, and quota error handling

## [1.6.0] — 2026-04-06

### Changed

- feat: bump version to 1.6.0

## [1.5.3] — 2026-04-05

### Changed

- chore: bump version to 1.5.3

## [1.5.2] — 2026-03-25

### Changed

- v1.5.2: clean up deprecated integration code

## [1.5.1] — 2026-02-28

### Changed

- Bump version to 1.5.1
