# PromptGuard Python SDK

`promptguard-sdk` on PyPI. Secures an application's LLM calls, either by
standing in the request path or by scanning calls the application still makes
itself.

The Node SDK is a deliberate mirror of this one. Where a term below names
something on the wire, the two SDKs are held to the same behaviour by the
guard contract, and the vocabulary must not drift either.

## Language

### Ways an application adopts the SDK

**Proxy mode**:
Using the `PromptGuard` client as a drop-in replacement for the provider's
client. PromptGuard is in the request path and forwards to the provider.
_Avoid_: proxy client mode, gateway mode. Note this is *not* the CLI's retired
sense of "proxy mode", which meant rewriting a base URL in source.

**Auto-instrumentation**:
Adopting the SDK by calling `init()`, so already-installed provider SDKs have
their calls scanned without the application changing how it calls them.
_Avoid_: runtime shim (the CLI's mechanism, not this one), monkey-patching
(that is how it works, not what it is), agent

**Framework integration**:
An adapter that carries scanning into a framework's own call path — LangChain,
CrewAI, LlamaIndex.
_Avoid_: plugin, middleware (that is one framework's word for it)

### Scanning

**Guard API**:
The endpoint that classifies content. Distinct from the proxy: it renders a
verdict on messages and never calls the provider.
_Avoid_: scan API, security API

**Scan**:
One Guard API call, about one set of messages, in one direction.
_Avoid_: check, inspection, guard call

**Direction**:
Which side of the provider call a scan covers — `input` before it, `output` on
what came back.
_Avoid_: request/response, pre/post, inbound/outbound

**Decision**:
The Guard API's verdict on a scan — allow, block, or redact. A missing or
unrecognised verdict is an error, never an implied allow.
_Avoid_: result, verdict, outcome

**Threat**:
One finding carried inside a decision, with its own type and confidence. A
decision may carry several.
_Avoid_: detection, violation, finding

**Enforce mode**:
The posture where a blocking decision stops the call. The default.
_Avoid_: blocking mode, strict mode

**Monitor mode**:
The posture where threats are recorded but no call is ever stopped.
_Avoid_: shadow mode — Shadow is a separate PromptGuard product (Shadow AI
devices, the Shadow fleet entitlement, the Shadow eval corpus) and those names
reach this repo through the generated API types. Also avoid: observe mode,
dry-run mode, passive mode.

**Fail-open**:
Allowing a call to proceed unscanned because the Guard API could not be
reached. The default, and a deliberate availability choice rather than a
fallback that happens to exist.
_Avoid_: fail-safe (ambiguous about which way it fails), degraded mode,
bypass

**Redaction**:
A decision that rewrites the outgoing messages instead of stopping the call.
When the call cannot be rewritten safely, enforce mode escalates it to a block.
_Avoid_: masking, sanitisation, scrubbing

### Staying in step with the Node SDK

**Guard contract**:
The shared, versioned file of cases that both SDKs must pass. A behaviour that
is not in the contract is not guaranteed to match across the two SDKs.
_Avoid_: contract tests (that is the suite that reads it), shared spec

**Contract lock**:
The recorded version and digest of the published contract a vendored copy came
from. Its job is to make an edit to either file alone fail.
_Avoid_: checksum file, manifest

**Generated API types**:
Types produced from the published OpenAPI spec by a scheduled sync, not
written by hand.
_Avoid_: schema types, models

### Auto-instrumentation vocabulary

**Applied patch**:
One provider SDK that `init()` has successfully hooked.
_Avoid_: patched SDK, instrumented SDK, hooked library. Note the two SDKs
currently disagree here — Python exposes `patched_sdks()`, Node exposes
`getAppliedPatches()` — and the guard contract does not cover this surface.

**Detected-unpatched library**:
A provider library that is installed and that this SDK knows how to patch, but
which `init()` did not hook. Reported by name, so an unhooked library is never
mistaken for one the application does not use.
_Avoid_: missed SDK, unsupported library, failed patch
