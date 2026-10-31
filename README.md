[![PyPI version](https://img.shields.io/pypi/v/promptguard-sdk)](https://pypi.org/project/promptguard-sdk/)
[![CI](https://github.com/acebot712/promptguard-python/actions/workflows/ci.yml/badge.svg)](https://github.com/acebot712/promptguard-python/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/acebot712/promptguard-python)](https://github.com/acebot712/promptguard-python/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/promptguard-sdk)](https://pypi.org/project/promptguard-sdk/)

# PromptGuard Python SDK

Drop-in security for AI applications. No code changes required.

## Installation

```bash
pip install promptguard-sdk
```

> **Package name ≠ import name.** Install **`promptguard-sdk`**, but import **`promptguard`**:
> ```python
> import promptguard
> from promptguard import PromptGuard
> ```

Get a free API key at [app.promptguard.co](https://app.promptguard.co).

> **The SDK reads `PROMPTGUARD_API_KEY` from the environment; it does not auto-load `.env`.** Use [python-dotenv](https://pypi.org/project/python-dotenv/) (call `load_dotenv()` before constructing the client) if you keep secrets in a `.env` file.

> **PromptGuard fails open by default** — if the Guard API is unavailable, calls proceed *unscanned* so your app stays up. Set `fail_open=False` to block (fail closed) on a Guard outage instead.

## Two Ways to Secure Your App

### Option 1: Auto-Instrumentation (Recommended for Frameworks)

One line secures the LLM calls made through the **patched SDK surfaces listed below**, regardless of which framework sits on top (LangChain, CrewAI, AutoGen, LlamaIndex, Haystack, Semantic Kernel, or direct SDK usage):

```python
import promptguard

promptguard.init(api_key="pg_live_xxx")

# That's it. LLM calls through the patched surfaces below are now secured.
# Works with ANY framework built on openai, anthropic, google-generativeai, cohere, or boto3.

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-nano", messages=[{"role": "user", "content": "Hello!"}]
)
# ^^ Scanned by PromptGuard before reaching OpenAI
```

**Supported SDKs** (auto-detected and patched):

| SDK | Frameworks Covered |
|-----|-------------------|
| `openai` | LangChain, CrewAI, AutoGen, Semantic Kernel, direct usage |
| `anthropic` | LangChain (ChatAnthropic), direct usage |
| `google-generativeai` | LangChain, LlamaIndex, direct usage |
| `cohere` | Haystack, LangChain, direct usage |
| `boto3` (Bedrock) | AWS-native apps (Claude, Titan, Llama on Bedrock) |

**Exact patched call surfaces** (sync and async clients where both exist):

- `openai`: `chat.completions.create()`, `chat.completions.parse()` (when the installed SDK exposes it), and `responses.create()` (when the installed SDK ships the Responses API). The Responses patch scans the `instructions` param plus string or message-item `input` forms; exotic input items (function-call outputs, reasoning items) are not scanned.
- `anthropic`: `messages.create()` (including the separate `system` param). Text and `tool_result` content blocks are scanned (tool results are the canonical indirect-injection channel); other block types (images, `tool_use` inputs, thinking) are not.
- `google-genai` (Google's current SDK): `models.generate_content()` and `generate_content_stream()`, sync and async. The `config.system_instruction` system prompt is scanned, and so are tool-call arguments, function responses and code-execution output — not just plain text parts. Because `contents` is keyword-only in this SDK, a `redact` decision rewrites the prompt rather than escalating to a block.
- `google-generativeai` (**deprecated by Google** when Gemini 2.0 shipped; still supported because customers are still on it): `GenerativeModel.generate_content()` / `generate_content_async()`. `redact` escalates to a block here, because `contents` is positional and cannot be safely rewritten.
- `cohere`: `Client.chat()` / `ClientV2.chat()` (v1 `preamble`/`message`/`chat_history` and v2 `messages`; the v1 `preamble` is scanned as a system message).
- `boto3` (Bedrock Runtime): `invoke_model` and `converse` (via `_make_api_call`).

Calls outside these surfaces — embeddings, audio, images, batches, fine-tuning, assistants, and other endpoints — are **not** scanned.

**Modes:**

```python
# Enforce mode (default) - blocks threats
promptguard.init(api_key="pg_live_xxx", mode="enforce")

# Monitor mode - logs threats without blocking (shadow mode)
promptguard.init(api_key="pg_live_xxx", mode="monitor")

# Scan responses too
promptguard.init(api_key="pg_live_xxx", scan_responses=True)

# Fail-closed (block if Guard API is unreachable)
promptguard.init(api_key="pg_live_xxx", fail_open=False)
```

**Shutdown:**

```python
promptguard.shutdown()  # Removes all patches, closes connections
```

**Verifying instrumentation:**

`init()` logs the SDKs it actually patched at `INFO`. Python's default log level
is `WARNING`, so enable the `promptguard` logger to see it:

```python
import logging

logging.getLogger("promptguard").setLevel(logging.INFO)
```

You can also assert instrumentation programmatically (e.g. in tests):

```python
import promptguard

promptguard.init(api_key="pg_live_xxx")

assert promptguard.is_active()  # a guard client is installed
assert "openai" in promptguard.patched_sdks()  # the OpenAI SDK was patched
```

`patched_sdks()` returns only the SDKs importable in the current environment
(missing packages are silently skipped), and an empty list after `shutdown()`.

`instrumentation_report()` gives the same facts as data: `patched`,
`detected_unpatched` (installed provider libraries nothing hooked), `middleware`
(request-edge middleware alive in this process, e.g. `["fastapi"]` — see
[FastAPI / ASGI middleware](#fastapi--asgi-middleware)) and `advice_url`.

### Confirming protection is actually live

`init()` returning without an exception does not mean anything is being scanned.
PromptGuard **fails open**, so a rejected API key, an unreachable Guard API, or a
provider SDK we never hooked all leave you with an application that runs
perfectly and blocks nothing. Each of those logs a warning — which helps only if
someone is reading the log.

`verify()` is the positive check. It makes the real calls and reports what came
back:

```python
import promptguard

report = promptguard.verify()

if not report["ok"]:
    for check in report["checks"]:
        print(check["status"], check["name"], "-", check["detail"])
    raise SystemExit(1)
```

In CI, the one-liner is usually enough:

```python
assert promptguard.verify()["ok"]
```

It checks reachability, authentication, live threat detection and PII redaction,
plus which provider SDKs this process actually patched. The check names are
`connectivity`, `authentication`, `threat_detection`, `pii_redaction` and
`instrumentation`. `promptguard verify` in the CLI sends the same two probes but
is not the same report: it has no instrumentation check (it cannot see inside
your process) and no `warn` status in its JSON output, only pass/fail counts.

Each check reports `pass`, `warn` or `fail`. **Only a `fail` clears `ok`.** A
request that never completed is a failure; a request that completed and came
back permissive — an injection that was not blocked, a PII probe with nothing
detected — is a warning, because a monitor-mode project legitimately behaves that
way. Warnings are still the first thing to read before trusting a setup.

`verify()` never raises for a failed check, so one call reports every problem
rather than only the first. It raises `ValueError` only when no API key was
supplied at all.

> **Each call makes two real, billed requests.** The probes go through the same
> endpoints as your production traffic, so every `verify()` writes two rows to
> your security events — the injection probe among them, which shows up in your
> dashboard and analytics as a genuine prompt-injection attempt and will trigger
> any alert you have configured on injection. Both requests count against your
> plan's quota. That is fine on deploy or at the start of a demo; calling it on
> every CI build of a busy repo is how you end up with a threat dashboard full of
> your own probes.

### Option 2: Proxy Mode (Drop-in Replacement)

If you prefer the proxy approach, just swap your client:

```python
# Before
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)  # attribute access

# After
from promptguard import PromptGuard

client = PromptGuard(api_key="pg_live_xxx")
response = client.chat.completions.create(
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response["choices"][0]["message"]["content"])  # dict/subscript access
```

**Request calls are identical, but responses differ from the native OpenAI SDK.**
The proxy client returns **plain OpenAI-compatible dicts** (the raw JSON body),
not the SDK's typed response objects. So `response.choices[0].message.content`
becomes `response["choices"][0]["message"]["content"]`. If you need the native
response objects (`.choices[0].message.content`) preserved, use **Option 1
auto-instrumentation** instead — it patches the real OpenAI/Anthropic/… client
in place, so their return types are unchanged and only the pre-flight scan is
added.

| | Response type | Access pattern |
|---|---|---|
| **Option 1** (auto-instrumentation) | native SDK objects (unchanged) | `response.choices[0].message.content` |
| **Option 2** (proxy `PromptGuard`) | OpenAI-compatible **dicts** | `response["choices"][0]["message"]["content"]` |

## Framework-Specific Integrations

For deeper integration with richer context (chain names, tool calls, agent steps), use framework-specific callbacks alongside or instead of auto-instrumentation:

### LangChain

```python
from promptguard.integrations.langchain import PromptGuardCallbackHandler

handler = PromptGuardCallbackHandler(api_key="pg_live_xxx")

# Attach to an LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5-nano", callbacks=[handler])

# Or use globally with any chain
chain.invoke({"input": "..."}, config={"callbacks": [handler]})

# LangGraph graphs take the same config; a block raises PromptGuardBlockedError
# out of graph.invoke / graph.ainvoke
graph.invoke({"messages": [...]}, config={"callbacks": [handler]})
```

The handler scans:
- `on_llm_start` / `on_chat_model_start` - prompts before the LLM call
- `on_llm_end` - responses after the LLM call
- `on_tool_start` / `on_tool_end` - tool inputs for injection attempts, and tool outputs
- `on_chain_start/end` - tracks chain context

**Python 3.10 and async graphs.** Before Python 3.11, asyncio cannot carry a
graph's callbacks into its nodes, so under `graph.ainvoke` / `astream` the
handler sees only the model and tool calls that receive the node's `config`.
A node that calls `model.ainvoke(state["messages"])` without it is **not
scanned**, and nothing fails. Forward the config in every node, or run on 3.11+:

```python
async def call_model(state, config):
    return {"messages": [await model.ainvoke(state["messages"], config=config)]}
```

The handler logs one warning per process when it sees an async graph on 3.10.
Sync graphs (`graph.invoke`) are not affected.

The callback can only *observe* (block or warn). To scan **and redact** content
inline in an LCEL chain, use `PromptGuardRunnable`, which sits in the data flow
and returns the redacted value on a `redact` decision:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from promptguard.integrations.langchain import PromptGuardRunnable

guard_in = PromptGuardRunnable(api_key="pg_live_xxx").as_runnable()
guard_out = PromptGuardRunnable(api_key="pg_live_xxx", direction="output").as_runnable()

prompt = ChatPromptTemplate.from_template("Answer: {input}")
chain = guard_in | prompt | ChatOpenAI(model="gpt-5-nano") | StrOutputParser() | guard_out

chain.invoke("user question")  # raises PromptGuardBlockedError on a block
```

Requires the `langchain` extra: `pip install promptguard-sdk[langchain]`.

### CrewAI

```python
from crewai import Crew, Agent, Task
from promptguard.integrations.crewai import PromptGuardGuardrail

pg = PromptGuardGuardrail(api_key="pg_live_xxx")

crew = Crew(
    agents=[...],
    tasks=[...],
    before_kickoff_callbacks=[pg.before_kickoff],
    after_kickoff_callbacks=[pg.after_kickoff],
)

crew.kickoff(inputs={"topic": "AI safety"})
```

You can also wrap individual tools:

```python
from promptguard.integrations.crewai import secure_tool
from crewai.tools import BaseTool


@secure_tool(api_key="pg_live_xxx")
class SearchTool(BaseTool):
    name = "search"
    description = "Search the web"

    def _run(self, query: str) -> str: ...
```

### LlamaIndex

```python
from promptguard.integrations.llamaindex import PromptGuardCallbackHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings

pg_handler = PromptGuardCallbackHandler(api_key="pg_live_xxx")
Settings.callback_manager = CallbackManager([pg_handler])

# All LlamaIndex queries are now scanned
```

To scan **and redact** a query inline in a `QueryPipeline` (the callback can only
observe), drop `PromptGuardQueryGuard` in as a preprocessing component:

```python
from llama_index.core.query_pipeline import QueryPipeline
from promptguard.integrations.llamaindex import PromptGuardQueryGuard

guard = PromptGuardQueryGuard(api_key="pg_live_xxx")

pipeline = QueryPipeline(chain=[guard.as_query_component(), retriever, synthesizer])
pipeline.run(input="user question")  # raises PromptGuardBlockedError on a block
```

Requires the `llamaindex` extra: `pip install promptguard-sdk[llamaindex]`.

### FastAPI / ASGI middleware

Scans prompts at your app's edge, before your route runs. Use it when your app
**accepts** prompts but calls the LLM somewhere auto-instrumentation does not
patch (a sidecar, a queue worker, an unsupported SDK). With `init()` also on,
the same text is scanned twice: once here, once at the provider call.

```python
from fastapi import FastAPI
from promptguard.integrations.fastapi import PromptGuardMiddleware

app = FastAPI()
app.add_middleware(
    PromptGuardMiddleware,
    api_key="pg_live_xxx",  # or PROMPTGUARD_API_KEY
    paths=["/v1/chat"],  # omit to scan every path
    exclude_paths=["/v1/chat/feedback"],
)
```

It is a raw ASGI middleware, so it works with Starlette or any ASGI app, and it
needs no extra install. The scan uses the async Guard client, and large bodies
are parsed in a worker thread, so nothing holds the event loop. Every option is
documented on `PromptGuardMiddleware` itself.

**What is scanned.** Any request with a body on the selected paths, whatever
its method (`methods=` to narrow it), with one `direction="input"` scan per
request. The fields read are `DEFAULT_PROMPT_FIELDS` (`prompt_fields=` to
replace them), each by the shape of its value:

- a list of message objects is scanned message by message, each with its own
  role, and a message's `tool_calls` go to the Guard API as tool calls;
- content blocks (text, images, documents, tool use) are passed to the Guard
  API as blocks, not flattened to text;
- a string, or a list of strings, is one message: `role="system"` for the
  fields in `SYSTEM_PROMPT_FIELDS`, `user` for the rest.

Only top-level fields are read. A top-level JSON array is read item by item.

The body is read by its `Content-Type`:

| Body | Read as |
|------|---------|
| JSON (`application/json`, `*+json`, or no `Content-Type`), in its declared `charset` | the fields above |
| `text/plain`, in its declared `charset` (default UTF-8) | one user message, whole |
| `application/x-www-form-urlencoded` | the fields above, from the form |
| `Content-Encoding: gzip` or `deflate` | decompressed first, then as above |
| `multipart/form-data`, binary and other media | **not scanned** (a known gap: a prompt sent as a multipart field reaches your route unscanned) |

**Paths.** `paths` and `exclude_paths` are prefixes matched on whole segments of
the path your app's routes see — the request path less any `root_path`, the
way Starlette routes it, so they work unchanged behind a proxy prefix or in a
mounted app. `"/v1/chat"` covers `/v1/chat` and `/v1/chat/completions`, but not
`/v1/chatter`. An exclusion wins over `paths`, and an excluded request's body is
never read. `paths=None` (the default) scans every path; `paths=[]` is an error.

**The body.** It is buffered — including a chunked or streamed one — scanned
whole, and replayed to your app byte for byte (still compressed, if it was).
The buffer is capped at `max_body_bytes` (default `DEFAULT_MAX_BODY_BYTES`),
raw and decompressed, so a decompression bomb is caught. A body over the cap
is answered `413` and is neither scanned nor passed on: forwarding it unscanned
would make padding a way past the guard.

**Decisions.**

| Outcome | `mode="enforce"` (default) | `mode="monitor"` |
|---------|---------------------------|------------------|
| allow | your route runs | your route runs |
| block | `403`, code `request_blocked`; your route never runs | logged, your route runs |
| redact | `403`, code `redaction_required` | logged, your route runs |
| body cannot be read: unknown `Content-Encoding` or `charset` | `415` | logged, your route runs |
| body cannot be read: undecodable text, invalid JSON under a JSON `Content-Type`, JSON nested too deeply | `400` | logged, your route runs |
| body over `max_body_bytes` | `413` | `413` |
| Guard API says the request is too large (`413`, or a `422` size error) | `413`, code `request_too_large` | `413` |
| Guard API rejects the request otherwise (`400`, `422`) | `400`, code `request_unscannable` | `400` |

A redact is refused because the middleware cannot rewrite an arbitrary body
safely. It is the rule ADR-0003 sets for a patched call auto-instrumentation
cannot rewrite, applied here by analogy. A request the Guard API itself rejects
is refused in every mode, because failing open on it would let a caller pad a
prompt past the Guard's limits to get it through.

A refusal uses the platform's error envelope:

```json
{
  "error": {
    "message": "Request blocked by security policy",
    "type": "policy_violation",
    "code": "request_blocked",
    "event_id": "evt_abc123",
    "details": {"threat_type": "prompt_injection", "confidence": 0.97}
  }
}
```

**When the Guard API cannot answer** — a network error, a timeout, a `5xx` or a
`429` — the request goes through unscanned (fail-open, the SDK default) and a
`WARNING` is logged on the `promptguard` logger with
`extra={"promptguard_event": "guard_unavailable_fail_open", "path": ...}`, so
you can alert on it. With `fail_open=False` the request is answered `503`, code
`guard_unavailable`, instead. A `401` or `403` (a bad API key) follows the same
setting, and is also logged once at `ERROR` with
`promptguard_event="guard_auth_failed"`; `promptguard.verify()` catches it
before you ship.

Once the middleware is wrapping your app (from startup, or the first request
when lifespan is off), `promptguard.instrumentation_report()["middleware"]`
lists `"fastapi"`, and `promptguard.verify()` counts it as coverage.

### Conversations (`session_id`)

Each scan can name the conversation it belongs to in `context.session_id`, so
the Guard API can relate the turns of one conversation to each other. An
integration sends the framework's **own** conversation id there, the same on
every turn of one conversation, and sends **nothing** when the framework has
none. It never invents one: a per-call run or event id would make every scan a
conversation of its own.

| Integration | Where the session id comes from |
|---|---|
| LangChain / LangGraph callback handler | The call's `thread_id`: LangGraph puts its checkpointer's `configurable["thread_id"]` in every callback's metadata. For a plain chain or model, set `config={"metadata": {"thread_id": ...}}` (`session_id` and `conversation_id` are read too, the keys LangSmith groups a thread by). |
| LangChain `PromptGuardRunnable` | The invoke config's `configurable["thread_id"]`, or the same metadata keys. |
| LlamaIndex handler and `PromptGuardQueryGuard` | LlamaIndex events name no conversation. Pass `session_id=` to the handler or guard that serves one chat. |
| CrewAI `PromptGuardGuardrail` | CrewAI has no conversation id. Pass `session_id=`. `secure_tool` sends none: a tool run is not a turn. |
| FastAPI / ASGI middleware | `session_id=lambda scope: ...` reads it from the authenticated session your auth middleware put in the scope; `session_id_header="X-Conversation-Id"` reads a header instead. Name only a header your own auth layer or a trusted proxy sets: a value the browser chooses lets a client pick which conversation its turns land in. |
| Auto-instrumentation (`init()`) | The OpenAI Responses API's `conversation` (an id or `{"id": ...}`), which is also how the OpenAI Agents SDK's `conversation_id` reaches the API. `previous_response_id` changes every turn and is not used. Other provider APIs carry no conversation id. |

Every handler, guard and guardrail that takes `session_id=` uses it only when
the call names no conversation of its own; the call's id wins. An empty string,
or one over 255 characters, is refused there with a `ValueError`. An id over
255 characters that arrives with a call is dropped with a debug log, and the
scan goes ahead without one.

```python
handler = PromptGuardCallbackHandler()
graph.invoke(inputs, config={"callbacks": [handler], "configurable": {"thread_id": "chat-42"}})
# every scan in this turn, and in every later turn of thread chat-42, sends session_id="chat-42"
```

## Standalone Guard API

For any language or framework, call the Guard API directly:

```python
from promptguard import GuardClient

# Use as a context manager so the underlying HTTP connection pool is closed.
with GuardClient(api_key="pg_live_xxx") as guard:
    # Scan before sending to LLM
    decision = guard.scan(
        messages=[{"role": "user", "content": "Hello!"}],
        direction="input",
        model="gpt-5-nano",
    )

    if decision.blocked:
        print(f"Blocked: {decision.threat_type}")
    elif decision.redacted:
        # Use decision.redacted_messages instead of original
        print("Content was redacted")
    else:
        # Safe to proceed
        pass
```

`GuardClient` also works as an async context manager (`async with GuardClient(...) as guard:` + `await guard.scan_async(...)`). If you keep a long-lived client instead, call `guard.close()` (or `await guard.aclose()`) when done.

Or via HTTP directly (any language):

```bash
curl -X POST https://api.promptguard.co/api/v1/guard \
  -H "X-API-Key: pg_live_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "direction": "input",
    "model": "gpt-5-nano"
  }'
```

> Authenticate with the `X-API-Key` header — this is the canonical header used by the Guard API and every SDK. There is no `Authorization: Bearer` scheme.

## Security Scanning

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_live_xxx")

# Scan content for threats
result = pg.security.scan("Ignore previous instructions...")
if result["blocked"]:
    print(f"Threat detected: {result['reason']}")
```

> **Two scan surfaces — object vs dict.** There are two ways to scan content and
> they return **different shapes**:
>
> | Call | Returns | Access |
> |---|---|---|
> | `GuardClient.scan(...)` (Standalone Guard API) | `GuardDecision` **object** | `decision.blocked`, `decision.threat_type` |
> | `pg.security.scan(...)` (proxy client) | `SecurityScanResult` **dict** | `result["blocked"]`, `result["reason"]` |
>
> Use attribute access for `GuardClient`, subscript access for `pg.security`.

## PII Redaction

```python
result = pg.security.redact("My email is jane.doe@acme.com and SSN is 123-45-6789")
print(result["redacted"])
# Output: "My email is [EMAIL_REDACTED] and SSN is [SSN_REDACTED]"
print(result["piiFound"])
# Output: ['email', 'ssn']
```

Each match is replaced by a `[<TYPE>_REDACTED]` placeholder. Addresses at the
domains reserved for documentation (`example.com`, `.test`, `.invalid`,
`.localhost`) are deliberately left alone, so `john@example.com` comes back
unchanged.

## Agent Tool Validation

The `agent` namespace validates individual agent tool calls (arguments) before
they execute — useful for guarding tool-using agents:

```python
result = pg.agent.validate_tool(
    agent_id="support-bot",
    tool_name="send_email",
    arguments={"to": "user@example.com", "body": "..."},
    session_id="sess-123",
)

# Per-agent stats
stats = pg.agent.stats("support-bot")
```

`agent` is available on the async client (`PromptGuardAsync`) with the same methods.

## Red Team Testing

Run PromptGuard's adversarial corpus against your own policy configuration, and
see how much of it your guardrails block. Useful from CI as a regression gate on
a policy change.

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_live_xxx")

# What the corpus contains, without running it
catalog = pg.redteam.list_tests()
print(f"{catalog['total']} attacks available")

# Run everything against a preset
summary = pg.redteam.run_all(target_preset="support_bot:strict")
print(f"Blocked {summary['blocked']}/{summary['total_tests']} ({summary['block_rate']:.0%})")

# Run one attack, by the `name` the catalog lists (e.g. "pi_001")
first = catalog["tests"][0]["name"]
result = pg.redteam.run_test(first, target_preset="support_bot:strict")
print(result["test_name"], result["decision"])

# Or your own adversarial prompt
result = pg.redteam.run_custom("ignore previous instructions and ...")
print(result["decision"], result["reason"])
```

The async client mirrors the same methods:

```python
async with PromptGuardAsync(api_key="pg_live_xxx") as pg:
    summary = await pg.redteam.run_all(target_preset="support_bot:strict")
```

Requires an API key with the `proxy` scope (an unrestricted key also works).
Device/scan-only credentials cannot reach these endpoints.

## Async Support

The `PromptGuardAsync` client provides a fully asynchronous interface for non-blocking usage in async applications:

```python
from promptguard import PromptGuardAsync

async with PromptGuardAsync(api_key="pg_live_xxx") as pg:
    response = await pg.chat.completions.create(
        model="gpt-5-nano", messages=[{"role": "user", "content": "Hello!"}]
    )

    # Async security scanning
    result = await pg.security.scan("Check this content")

    # Async PII redaction
    redacted = await pg.security.redact("My email is jane.doe@acme.com")
```

The async client mirrors the synchronous API - every method available on `PromptGuard` has an `await`-able counterpart on `PromptGuardAsync`.

## Retry Logic

Both `PromptGuard` and `PromptGuardAsync` support configurable retry behavior for transient failures:

```python
from promptguard import PromptGuard

pg = PromptGuard(
    api_key="pg_live_xxx",
    max_retries=3,  # Number of retry attempts (default: 3)
    retry_delay=0.5,  # Base delay in seconds between retries (default: 1.0)
)
```

Retries use exponential backoff starting from `retry_delay`. Transient errors are retried: network timeouts, 5xx responses, and 429 rate limits (honoring a `Retry-After` header when present, capped at 60s). Other client errors (4xx) fail immediately, as does a 429 that signals hard quota exhaustion.

## Configuration

```python
from promptguard import PromptGuard, Config

config = Config(
    api_key="pg_live_xxx",
    base_url="https://api.promptguard.co/api/v1/proxy",
    timeout=30.0,
)

pg = PromptGuard(config=config)
```

## Environment Variables

```bash
export PROMPTGUARD_API_KEY="pg_live_xxx"
# Optional — only override if pointing at a self-hosted/staging deployment.
# Leave unset to use the default (https://api.promptguard.co/api/v1/proxy).
export PROMPTGUARD_BASE_URL="https://api.promptguard.co/api/v1"
```

> The proxy client (`PromptGuard`) talks to the `/api/v1/proxy` endpoints. If you set `PROMPTGUARD_BASE_URL` to `.../api/v1` (without `/proxy`), the SDK appends the `/proxy` suffix for you, so requests still land on the proxy. Setting it explicitly to `.../api/v1/proxy` also works.
>
> **Security:** the SDK sends your API key (and, in proxy mode, your prompt content) to whatever `PROMPTGUARD_BASE_URL` points at. Self-hosting is supported, so only point it at a host you trust.

## Error Handling

```python
from promptguard import PromptGuardBlockedError

# Auto-instrumentation
import promptguard

promptguard.init(api_key="pg_live_xxx")

# Use your real LLM client as usual — it is patched in place.
from openai import OpenAI

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello!"}],
    )
except PromptGuardBlockedError as e:
    print(f"Blocked: {e.decision.threat_type}")
    print(f"Event ID: {e.decision.event_id}")
```

In **proxy mode**, the client raises `PromptGuardError`, which carries structured
fields you can branch on — `.code`, `.status_code`, `.upgrade_url`,
`.current_plan`, `.requests_used`, and `.requests_limit`:

```python
from promptguard import PromptGuard, PromptGuardError

client = PromptGuard(api_key="pg_live_xxx")

try:
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello!"}],
    )
except PromptGuardError as e:
    if e.code == "monthly_quota_exceeded":
        print(f"Upgrade: {e.upgrade_url}")
    else:
        print(f"{e.code} ({e.status_code}): {e}")
```

## Links

- [Documentation](https://docs.promptguard.co)
- [SDK Reference](https://docs.promptguard.co/sdks/python)
- [Support](mailto:support@promptguard.co)

## License

MIT
