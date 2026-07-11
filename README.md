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
    model="gpt-5-nano",
    messages=[{"role": "user", "content": "Hello!"}]
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
- `google-generativeai`: `GenerativeModel.generate_content()` / `generate_content_async()`.
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

assert promptguard.is_active()               # a guard client is installed
assert "openai" in promptguard.patched_sdks()  # the OpenAI SDK was patched
```

`patched_sdks()` returns only the SDKs importable in the current environment
(missing packages are silently skipped), and an empty list after `shutdown()`.

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
```

The handler scans:
- `on_llm_start` / `on_chat_model_start` - prompts before the LLM call
- `on_llm_end` - responses after the LLM call
- `on_tool_start` - tool inputs for injection attempts
- `on_chain_start/end` - tracks chain context

### CrewAI

```python
from crewai import Crew, Agent, Task
from promptguard.integrations.crewai import PromptGuardGuardrail

pg = PromptGuardGuardrail(api_key="pg_live_xxx")

crew = Crew(
    agents=[...],
    tasks=[...],
    before_kickoff=pg.before_kickoff,
    after_kickoff=pg.after_kickoff,
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

    def _run(self, query: str) -> str:
        ...
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

## Standalone Guard API

For any language or framework, call the Guard API directly:

```python
from promptguard import GuardClient

guard = GuardClient(api_key="pg_live_xxx")

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
result = pg.security.redact(
    "My email is john@example.com and SSN is 123-45-6789"
)
print(result["redacted"])
# Output: "My email is [EMAIL] and SSN is [SSN]"
```

## Web Scraping

The proxy client exposes a `scrape` namespace for fetching and extracting page
content through PromptGuard (responses are plain dicts):

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_live_xxx")

# Single URL — returns a dict with the extracted content
result = pg.scrape.url("https://example.com", render_js=False, extract_text=True)

# Batch
results = pg.scrape.batch(["https://a.com", "https://b.com"])
```

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

Both `scrape` and `agent` are available on the async client (`PromptGuardAsync`)
with the same methods.

## Red Team Testing

> **Preview / internal endpoint.** The `redteam` namespace targets the
> `/api/v1/proxy/internal/redteam` path. It is a supported but preview-tier
> surface intended for security testing; availability and response shapes may
> change ahead of the other proxy namespaces, and access may be gated by plan.

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_live_xxx")

# Run the autonomous red team agent (LLM-powered mutation)
report = pg.redteam.run_autonomous(
    budget=200,
    target_preset="support_bot:strict",
)
print(f"Grade: {report['grade']}, Bypass rate: {report['bypass_rate']:.0%}")

# Get Attack Intelligence stats
stats = pg.redteam.intelligence_stats()
print(f"Total patterns: {stats['total_patterns']}")
```

The async client mirrors the same methods:

```python
async with PromptGuardAsync(api_key="pg_live_xxx") as pg:
    report = await pg.redteam.run_autonomous(budget=200)
    stats = await pg.redteam.intelligence_stats()
```

## Async Support

The `PromptGuardAsync` client provides a fully asynchronous interface for non-blocking usage in async applications:

```python
from promptguard import PromptGuardAsync

async with PromptGuardAsync(api_key="pg_live_xxx") as pg:
    response = await pg.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Async security scanning
    result = await pg.security.scan("Check this content")

    # Async PII redaction
    redacted = await pg.security.redact("My email is john@example.com")
```

The async client mirrors the synchronous API - every method available on `PromptGuard` has an `await`-able counterpart on `PromptGuardAsync`.

## Retry Logic

Both `PromptGuard` and `PromptGuardAsync` support configurable retry behavior for transient failures:

```python
from promptguard import PromptGuard

pg = PromptGuard(
    api_key="pg_live_xxx",
    max_retries=3,        # Number of retry attempts (default: 3)
    retry_delay=0.5,      # Base delay in seconds between retries (default: 1.0)
)
```

Retries use exponential backoff starting from `retry_delay`. Transient errors are retried: network timeouts, 5xx responses, and 429 rate limits (honoring a `Retry-After` header when present, capped at 60s). Other client errors (4xx) fail immediately, as does a 429 that signals hard quota exhaustion.

## Embeddings

Scan and secure embedding requests through the proxy:

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_live_xxx")

response = pg.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog",
)
# Proxy responses are returned as plain dicts (OpenAI-compatible JSON shape).
print(response["data"][0]["embedding"][:5])
```

Batch embedding requests are also supported:

```python
response = pg.embeddings.create(
    model="text-embedding-3-small",
    input=["First document", "Second document", "Third document"],
)
for item in response["data"]:
    print(f"Index {item['index']}: {len(item['embedding'])} dimensions")
```

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

## Links

- [Documentation](https://docs.promptguard.co)
- [SDK Reference](https://docs.promptguard.co/sdks/python)
- [Support](mailto:support@promptguard.co)

## License

MIT
