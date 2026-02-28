# PromptGuard Python SDK

Drop-in security for AI applications. No code changes required.

## Installation

```bash
pip install promptguard-sdk
```

## Two Ways to Secure Your App

### Option 1: Auto-Instrumentation (Recommended for Frameworks)

One line secures **every LLM call** in your application, regardless of which framework you use (LangChain, CrewAI, AutoGen, LlamaIndex, Haystack, Semantic Kernel, or direct SDK usage):

```python
import promptguard
promptguard.init(api_key="pg_xxx")

# That's it. Every LLM call is now secured.
# Works with ANY framework built on openai, anthropic, google-generativeai, cohere, or boto3.

from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
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

**Modes:**

```python
# Enforce mode (default) — blocks threats
promptguard.init(api_key="pg_xxx", mode="enforce")

# Monitor mode — logs threats without blocking (shadow mode)
promptguard.init(api_key="pg_xxx", mode="monitor")

# Scan responses too
promptguard.init(api_key="pg_xxx", scan_responses=True)

# Fail-closed (block if Guard API is unreachable)
promptguard.init(api_key="pg_xxx", fail_open=False)
```

**Shutdown:**

```python
promptguard.shutdown()  # Removes all patches, closes connections
```

### Option 2: Proxy Mode (Drop-in Replacement)

If you prefer the proxy approach, just swap your client:

```python
# Before
from openai import OpenAI
client = OpenAI()

# After
from promptguard import PromptGuard
client = PromptGuard(api_key="pg_xxx")

# Your existing code works unchanged!
```

## Framework-Specific Integrations

For deeper integration with richer context (chain names, tool calls, agent steps), use framework-specific callbacks alongside or instead of auto-instrumentation:

### LangChain

```python
from promptguard.integrations.langchain import PromptGuardCallbackHandler

handler = PromptGuardCallbackHandler(api_key="pg_xxx")

# Attach to an LLM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

# Or use globally with any chain
chain.invoke({"input": "..."}, config={"callbacks": [handler]})
```

The handler scans:
- `on_llm_start` / `on_chat_model_start` — prompts before the LLM call
- `on_llm_end` — responses after the LLM call
- `on_tool_start` — tool inputs for injection attempts
- `on_chain_start/end` — tracks chain context

### CrewAI

```python
from crewai import Crew, Agent, Task
from promptguard.integrations.crewai import PromptGuardGuardrail

pg = PromptGuardGuardrail(api_key="pg_xxx")

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

@secure_tool(api_key="pg_xxx")
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

pg_handler = PromptGuardCallbackHandler(api_key="pg_xxx")
Settings.callback_manager = CallbackManager([pg_handler])

# All LlamaIndex queries are now scanned
```

## Standalone Guard API

For any language or framework, call the Guard API directly:

```python
from promptguard import GuardClient

guard = GuardClient(api_key="pg_xxx")

# Scan before sending to LLM
decision = guard.scan(
    messages=[{"role": "user", "content": "Hello!"}],
    direction="input",
    model="gpt-4o",
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
  -H "Authorization: Bearer pg_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "direction": "input",
    "model": "gpt-4o"
  }'
```

## Security Scanning

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_xxx")

# Scan content for threats
result = pg.security.scan("Ignore previous instructions...")
if result["blocked"]:
    print(f"Threat detected: {result['reason']}")
```

## PII Redaction

```python
result = pg.security.redact(
    "My email is john@example.com and SSN is 123-45-6789"
)
print(result["redacted"])
# Output: "My email is [EMAIL] and SSN is [SSN]"
```

## Red Team Testing

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_xxx")

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
async with PromptGuardAsync(api_key="pg_xxx") as pg:
    report = await pg.redteam.run_autonomous(budget=200)
    stats = await pg.redteam.intelligence_stats()
```

## Async Support

The `PromptGuardAsync` client provides a fully asynchronous interface for non-blocking usage in async applications:

```python
from promptguard import PromptGuardAsync

async with PromptGuardAsync(api_key="pg_xxx") as pg:
    response = await pg.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Async security scanning
    result = await pg.security.scan("Check this content")

    # Async PII redaction
    redacted = await pg.security.redact("My email is john@example.com")
```

The async client mirrors the synchronous API — every method available on `PromptGuard` has an `await`-able counterpart on `PromptGuardAsync`.

## Retry Logic

Both `PromptGuard` and `PromptGuardAsync` support configurable retry behavior for transient failures:

```python
from promptguard import PromptGuard

pg = PromptGuard(
    api_key="pg_xxx",
    max_retries=3,        # Number of retry attempts (default: 2)
    retry_delay=0.5,      # Base delay in seconds between retries (default: 0.25)
)
```

Retries use exponential backoff starting from `retry_delay`. Only transient errors (network timeouts, 5xx responses) are retried — client errors (4xx) fail immediately.

## Embeddings

Scan and secure embedding requests through the proxy:

```python
from promptguard import PromptGuard

pg = PromptGuard(api_key="pg_xxx")

response = pg.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog",
)
print(response.data[0].embedding[:5])
```

Batch embedding requests are also supported:

```python
response = pg.embeddings.create(
    model="text-embedding-3-small",
    input=["First document", "Second document", "Third document"],
)
for item in response.data:
    print(f"Index {item.index}: {len(item.embedding)} dimensions")
```

## Configuration

```python
from promptguard import PromptGuard, Config

config = Config(
    api_key="pg_xxx",
    base_url="https://api.promptguard.co/api/v1/proxy",
    enable_caching=True,
    enable_security_scan=True,
    timeout=30.0,
)

pg = PromptGuard(config=config)
```

## Environment Variables

```bash
export PROMPTGUARD_API_KEY="pg_xxx"
export PROMPTGUARD_BASE_URL="https://api.promptguard.co/api/v1"
```

## Error Handling

```python
from promptguard import PromptGuard, PromptGuardBlockedError

# Auto-instrumentation
import promptguard
promptguard.init(api_key="pg_xxx")

try:
    response = client.chat.completions.create(...)
except PromptGuardBlockedError as e:
    print(f"Blocked: {e.decision.threat_type}")
    print(f"Event ID: {e.decision.event_id}")
```

## License

MIT
