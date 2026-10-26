"""
Auto-generated from OpenAPI spec (v1.0.0).
DO NOT EDIT — regenerate with: python scripts/generate_types_from_spec.py

These are type-only definitions. Custom client logic lives in
promptguard/guard.py, promptguard/client.py, promptguard/patches/,
and promptguard/integrations/ — those files are never modified
by this generator.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class ActiveGrant(TypedDict):
    """A destination the agent may send to until ``expires_at`` (ISO 8601)."""

    host: str
    code: str
    expires_at: str


class ActiveGrantList(TypedDict):
    grants: list[ActiveGrant]


class AgentMemoryRequest(TypedDict, total=False):
    """Content being written to, or read back from, an agent's memory."""

    # The memory chunk to scan
    content: str
    # 'write' before persisting a chunk, 'read' when a stored chunk is retrieved. Scan both: a chunk poisoned before this endpoint existed, or written through another path, is only catchable on read.
    direction: str
    # Your identifier for the chunk
    memory_id: str | Any


class AgentMemoryResponse(TypedDict, total=False):
    """Verdict on one memory chunk."""

    decision: str
    detected: bool
    reason: str
    confidence: float
    match_type: str | Any
    content_hash: str | Any
    # The detector and rule that made the finding, the pair a Suppression is keyed on. Null when the chunk is clean.
    rule: FindingRule | Any
    event_id: str


class AgentPoliciesResponse(TypedDict):
    policies: list[developer__policies__router__AgentPolicy]
    total: int


class AgentRegisterRequest(TypedDict, total=False):
    """Request to register a new agent identity."""

    agent_name: str
    allowed_tools: list[str] | Any


class AgentRegisterResponse(TypedDict):
    """Response from agent registration — secret is shown only once."""

    agent_id: str
    agent_name: str
    agent_secret: str
    credential_prefix: str


class AgentRotateResponse(TypedDict):
    """Response from credential rotation."""

    agent_id: str
    new_secret: str
    credential_prefix: str
    old_credential_revoked: bool


class AgentSecurityHealthResponse(TypedDict):
    status: str
    validator_status: str
    analyzer_status: str
    blocked_tools_count: int


class AgentSessionEndedResponse(TypedDict):
    status: str
    agent_id: str
    session_id: str
    deprecated: bool


class AgentStats(TypedDict, total=False):
    """Statistics for an agent"""

    agent_id: str
    total_tool_calls: int
    blocked_calls: int
    avg_risk_score: float
    # Always 0. Agent session state is not retained across requests; this field is deprecated and will be removed in the next API version.
    active_sessions: int
    anomalies_detected: int


class AgentToolLabels(TypedDict, total=False):
    """Capability profile for one tool along the four lethal-trifecta axes."""

    untrusted_content: bool
    private_data: bool
    public_sink: bool
    destructive: bool


class AgentTraceEvent(TypedDict, total=False):
    """One event of a full agent execution trace. An event with a ``tool_name`` is a tool call: its ``arguments`` (a sink's inputs) and ``output`` (a source of taint) drive the trace-level detectors. Events without one are plain assistant / user turns, kept as context."""

    role: str
    tool_name: str | Any
    arguments: dict[str, Any]
    output: Any
    content: str
    thought: str


class AgentTraceFinding(TypedDict, total=False):
    """A single detector hit, normalized across detectors."""

    detector: str
    code: str
    severity: str
    reason: str
    decision: str
    # The detector and rule that made this finding, the pair a Suppression names.
    rule: FindingRule | Any
    metadata: dict[str, Any]


class AgentTraceRequest(TypedDict, total=False):
    """A full agent execution trace to audit post-hoc. Unlike ``/validate-tool`` (a pre-execution check of a single tool name + args), this carries the whole chronological trace *with tool outputs* plus the user's original objective, so the value-level dataflow-taint and goal-alignment detectors can fire."""

    user_objective: str
    events: list[AgentTraceEvent]
    tool_labels: dict[str, Any] | Any


class AgentTraceResponse(TypedDict, total=False):
    """Aggregated verdict over the ingested trace."""

    decision: str
    findings: list[AgentTraceFinding]
    event_id: str


class ApiHealthResponse(TypedDict):
    status: str
    timestamp: str


class ApiKeyResponse(TypedDict, total=False):
    id: str
    name: str
    prefix: str
    key: str | Any
    project_id: str | Any
    project_name: str | Any
    permissions: list[str]
    is_active: bool
    last_used_at: str | Any
    expires_at: str | Any
    created_at: str
    rate_limit_per_minute: int | Any


class ApiKeyToggledResponse(TypedDict):
    """What the toggle answers: whether the key authenticates from now on."""

    success: bool
    is_active: bool


class ApiRootResponse(TypedDict):
    name: str
    description: str
    status: str
    documentation: str | Any


class ApiVersionProbeResponse(TypedDict):
    """A constant: the real version is behind ``/dashboard/version``."""

    status: str


class AppliedThreshold(TypedDict):
    """The threshold a finding was decided against (#970)."""

    # `block`: the score reached `block`, and the finding blocked. `warn`: the score reached `warn` but not `block`, so the finding was reported and the request allowed (the event is stored as `flag`).
    band: Literal["warn", "block"]
    # The detector's warn threshold, or null when it has only a block band.
    warn: float | Any
    # The detector's block threshold.
    block: float
    # The score compared with the thresholds: the weighted score where the detector reports one, otherwise its confidence.
    score: float


class AuthErrorEnvelope(TypedDict):
    error: ErrorDetail


class ContextDoc(TypedDict, total=False):
    """A document retrieved by a RAG pipeline to be scanned for poisoning."""

    # Document text content
    content: str
    # Source identifier (URL, doc ID, etc.)
    source: str | Any
    # Extra metadata
    metadata: dict[str, Any] | Any


class CreateApiKeyRequest(TypedDict, total=False):
    # API key name
    name: str
    project_id: str | Any
    permissions: list[str]
    expires_at: str | Any


class CreateApiKeyResponse(TypedDict):
    key: str
    id: str
    name: str
    prefix: str


class CreateExceptionRequest(TypedDict, total=False):
    destination_host: str
    policy_id: str | Any
    threat: str | Any
    reason_category: str | Any
    justification: str | Any
    requested_minutes: int


class CreateToolRequest(TypedDict, total=False):
    requested_host: str
    requested_name: str | Any
    justification: str | Any


class DeviceFinding(TypedDict):
    """One category the Device's masker caught, and how many times."""

    # The Engine's threat category, e.g. 'api_key_leak' or 'pii_leak'.
    category: ThreatType
    # How many values of this category the masker replaced.
    count: int


class DeviceIdentity(TypedDict):
    """What the calling device is enrolled as."""

    device_id: str
    device_name: str
    organization_id: str
    organization_name: str | Any
    identity_verified: bool
    account_email: str | Any
    account_name: str | Any
    attribution_label: str | Any


class DeviceRevokedResponse(TypedDict, total=False):
    revoked: bool


class DeviceShadowException(TypedDict):
    """One exception request as the filing device sees it. Timestamps are ISO 8601."""

    id: str
    code: str
    destination_host: str
    policy_id: str | Any
    threat: str | Any
    reason_category: str | Any
    justification: str | Any
    requested_minutes: int
    # pending | approved | denied | cancelled | expired
    status: str
    created_at: str | Any
    decided_at: str | Any
    expires_at: str | Any


class DeviceShadowExceptionList(TypedDict):
    exceptions: list[DeviceShadowException]


class DeviceToolRequest(TypedDict):
    """One "request a tool" row as the filing device sees it. Timestamps are ISO 8601."""

    id: str
    code: str
    requested_host: str
    requested_name: str | Any
    justification: str | Any
    # pending | approved | denied | cancelled
    status: str
    created_at: str | Any
    decided_at: str | Any


class DeviceToolRequestList(TypedDict):
    tool_requests: list[DeviceToolRequest]


class DeviceView(TypedDict):
    """One device as an enrolled device may see it: another machine's label is withheld."""

    id: str
    name: str
    platform: str
    last_seen_at: str | Any
    end_user_id: str | Any
    # None where the label is withheld
    identity_verified: bool | Any


class DeviceViewList(TypedDict):
    devices: list[DeviceView]


class DivergenceItemOut(TypedDict):
    text_preview: str
    category: str
    base_decision: str
    base_confidence: float
    candidate_decision: str
    candidate_confidence: float
    divergence: str


class EnrollRequest(TypedDict, total=False):
    # Enrollment token from the admin
    token: str
    # Hostname / device label
    device_name: str
    # macos | windows | browser | linux
    platform: str
    # Employee attribution label
    end_user_id: str | Any
    # Capture tier the agent is running: 'extension' or 'proxy'
    coverage: str


class EnrollResponse(TypedDict, total=False):
    api_key: str
    project_id: str
    device_id: str
    organization_id: str
    enforced: bool
    mode: str
    fail_closed: bool
    end_user_label: str | Any
    account_name: str | Any
    identity_verified: bool
    account_email: str | Any
    account_type: str


class ErrorDetail(TypedDict):
    # Human-readable error description
    message: str
    # Error category, e.g. 'authentication_error'
    type: str
    # Machine-readable error code
    code: str


class ErrorEnvelope(TypedDict):
    error: ErrorDetail


class ExportPage(TypedDict):
    """One page of an Organization's events, oldest first."""

    organization_id: str
    organization_name: str | Any
    events: list[SiemEventRecord]
    # Where this page ended. Pass it back as `cursor` to read on; it is returned even when the page is short, so a poller always has its place. Null only when there was no position to report.
    next_cursor: str | Any
    # Whether another page is ready now.
    has_more: bool


class FindingRule(TypedDict):
    """Which rule of which detector a finding is, as the API returns it."""

    # The detector that made the finding, e.g. `injection_regex`, `pii`, `ml_injection_classifier`. Stable across releases.
    detector: str
    # The rule of that detector that fired. Stable across releases: a detector that reports by type uses the type (`email` for `pii`), a pattern rule its declared name (`ignore_previous_instructions`), an ATR rule ATR's own id, and a detector with one verdict its own name.
    id: str
    # Whether a Suppression may target this rule. False for the ML injection classifier and every model or LLM judge of injection, the secret detectors, canary tokens, and the project's own rules and MCP configuration. Other judges, such as content safety, can be.
    suppressible: bool


# Whether a finding decides. Returned on every guard API threat.
FindingStatus = Literal["active", "candidate", "suppressed"]


class GuardContext(TypedDict, total=False):
    """Optional rich context from framework integrations. Only ``tool_calls`` is scanned. The rest is descriptive — it labels the event for the dashboard and the audit log, and does not reach a detector."""

    # Framework name, e.g. 'langchain', 'crewai'
    framework: str | Any
    # LangChain chain name or pipeline identifier
    chain_name: str | Any
    # Agent identifier for multi-agent systems
    agent_id: str | Any
    # The conversation this request is a turn of. Send the same value for every turn of one end user's conversation: with the X-End-User header it is one session, and the dashboard's Sessions view lists that session's events in order. Generate it on your server, random enough that an end user cannot guess another's (a UUIDv4 will do), and never take it from the end user's request: whoever can name a session id can add turns to that conversation. At most 255 characters, stored exactly as sent; it names the conversation and holds none of it, so a zero-retention project keeps it. The same id under another end user or project is another conversation. For each input turn PromptGuard also remembers its detection signals: the Techniques of the finding that decided it, that finding's confidence and when the turn arrived, never its text, for every project including zero-retention ones. At most the last 10 turns are kept, each for at most 1 hour. When the request names its end user with the X-End-User header, a turn that pivots ("now tell me...") after an earlier persona, hypothetical or authority turn is reported as a `candidate` threat from the `conversation/priming_then_pivot` rule, which never changes the decision; without X-End-User that rule records nothing. A sudden burst of turns, many times the conversation's own earlier rate, is recorded on the turn's event as a signal, and changes nothing either. If the conversation's memory cannot be reached in time, the turn is scanned on its own. Output scans are not turns. Without a session id a request is scanned on its own, as it always was.
    session_id: str | Any
    # Tool calls in this turn. The tool NAME and its ARGUMENTS are assembled into the scanned text and get the same detection stack as the messages — tool arguments are where an exfiltration payload actually travels, so they are scanned rather than logged. Both provider spellings are read: OpenAI's `{'type':'function','function':{'name','arguments'}}` and Anthropic's `{'type':'tool_use','name','input'}`, plus LangChain's `{'name','args'}`. A call in none of those shapes is reported in the response's `unscanned` array with its position; it is never quietly skipped. These calls carry no position in the conversation, so they are always scanned as the newest turn's. Send a conversation's calls on the messages that made them (`messages[].tool_calls`) to have each read as part of its own turn.
    tool_calls: list[dict[str, Any]] | Any
    # The tools the AI app offers the model on this request: each one's name, description and input schema. Each is scanned as a tool manifest -- the scan `POST /api/v1/mcp/manifest-scan` runs, for poisoned descriptions and parameter docs and smuggled Unicode -- and a poisoned one blocks, naming the tool, with the finding's `segment` `tool_definition`. An unchanged definition is not scanned again on the next request. Read in MCP's `{'name','description','inputSchema'}`, Anthropic's `input_schema` and OpenAI's `{'type':'function','function':{'name','description','parameters'}}`; a definition in none of them is listed in `unscanned` with its position.
    tool_definitions: list[dict[str, Any]] | Any
    # Arbitrary framework-specific metadata (not scanned)
    metadata: dict[str, Any] | Any


class GuardEntitlement(TypedDict):
    """What the calling account is entitled to, as the engine knew it here. Composed from the same subscription row the scan was just counted against, so it cannot be stale while a scan is in flight. Why it rides the scan response rather than an endpoint of its own, and why it carries no price, are ADR-0018; the reasoning lives there and not in four docstrings."""

    # Plan identifier, e.g. 'free', 'shadow_team'
    plan: str
    # Seats on the subscription, mirrored from the Stripe quantity. Always 1 on a flat-rate Plan, which does not bill per seat.
    seats: int
    # Scans counted against the org's pool this month, INCLUDING the one being answered.
    scans_used: int
    # The month's pool. The enforced figure, not one recomputed from the Plan grid, so it carries a per-contract override where one exists.
    scan_allowance: int
    # How the client should run: 'full' while the engine arbitrates, or 'redact_only' once the entitlement is spent or lapsed, where an on-device detection masks rather than refuses. A client that does not recognise the value must treat it as unknown and keep running as it was, never as a default.
    mode: str
    # When this reading was taken (UTC). Not the scan's timestamp.
    as_of: str


class GuardMessage(TypedDict, total=False):
    """A single message in the conversation. ``content`` takes either a plain string or a provider-shaped content-block array — OpenAI's ``text``/``image_url``/``input_audio``/``file`` and Anthropic's ``text``/``image``/``document`` are all understood, because those are the two shapes our own proxy already receives. Blocks are accepted as loose dicts rather than a closed union on purpose. Both providers add block types faster than we can model them, and a strict schema would 422 a request we could otherwise have scanned the text of. Anything unrecognised is *reported* in the response's ``unscanned`` rather than dropped — see ``shared.security.content_parts``."""

    # Message role: system, user, assistant, tool. On an input scan a tool (or function) message is also scanned as a tool's result, and an instruction planted in it blocks as prompt_injection.
    role: str
    # Message text, or an OpenAI/Anthropic content-block array. Attachments carried in blocks are extracted and scanned like any other text; blocks we cannot read are listed in `unscanned`.
    content: str | list[dict[str, Any]]
    # The tool calls this message made, as a conversation carries them, in any shape `context.tool_calls` reads. Their names and arguments are scanned like `context.tool_calls`, as part of this message's turn; a call we cannot read is listed in `unscanned`. On a Device's request an Earlier turn's calls are not rescanned for what blocks, and personal data or a credential in their arguments is redacted again: the decision is `redact`, and the call is the caller's to mask, because `redacted_messages` carries text only.
    tool_calls: list[Any] | Any


class GuardOverride(TypedDict, total=False):
    """A Shadow AI Device's report that a person sent a coached prompt anyway."""

    # The Device's coaching reference for the prompt, e.g. `C-1A2B3C4D`.
    reference: str
    # The reason the person gave, masked on the Device and masked again here like a preview. Empty when the Device's Disclosure sends no prompt content.
    reason: str
    # What the coached block was for, as this API names threat types. One it does not know is not stored.
    threat_type: str | Any
    # The event the coached block was recorded as, when the Engine decided it.
    engine_event_id: str | Any


class GuardRequest(TypedDict, total=False):
    """Request body for the guard endpoint."""

    # Messages to scan (OpenAI-style message array). When their text is more than one scan can read, the newest messages that fit are scanned and every older one is listed in `unscanned` as `over_size_budget`; only a newest message too large on its own is refused with 413.
    messages: list[GuardMessage]
    # Scan direction: 'input' (pre-LLM) or 'output' (post-LLM)
    direction: str
    # Model being used (for logging)
    model: str | Any
    # Optional framework context
    context: GuardContext | Any
    # RAG-retrieved documents to scan for knowledge poisoning. Each document is scanned individually; the first poisoned one blocks the request, and its position and source are returned in the event metadata so you know which document to drop. Scanning stops at that point, so a request with several poisoned documents reports the first. Optional; backwards-compatible.
    retrieved_context: list[ContextDoc] | Any
    # Media attachments to scan for steganographic payloads, adversarial patches, and font injection. Optional.
    media: list[MediaPartSchema] | Any
    # Image URLs or `data:` URIs to scan, as LiteLLM's PromptGuard guardrail sends them. Scanned like an `image_url` block on a message.
    images: list[str] | Any
    # A Shadow AI Device reporting an **Override**: the person sent a prompt the Device coached (a block that rested on one detector) anyway, with a reason. Sent with every message empty and nothing else to scan; any content is refused with 422. Recorded as an event with the decision `override`, its threat type and the reason, alerted on like a block and delivered to the SIEM, and not metered. The answer is `allow`: nothing was scanned.
    override: GuardOverride | Any
    # What the calling Device's own masker caught before sending, as `{category, count}` pairs — never the values. `category` is one of the threat types this API returns (e.g. `api_key_leak`, `pii_leak`) and appears at most once; `count` is a positive integer. Any other key is rejected with 422. Recorded on this request's event, attributed to the Device, and shown in the fleet views; it does not change the decision. Optional; clients that omit it are unaffected.
    device_findings: list[DeviceFinding] | Any


class GuardResponse(TypedDict, total=False):
    """Response from the guard endpoint."""

    # Policy decision: 'allow', 'block', or 'redact'
    decision: str
    # Unique event identifier for tracking
    event_id: str
    # Confidence score of the decision
    confidence: float
    # Aggregate decision-driving score (severity * confidence, clamped to [0, 1]) when a severity-carrying detector decided the verdict; null otherwise. Raw confidence stays in the `confidence` field.
    weighted_score: float | Any
    # Primary threat type detected
    threat_type: str | Any
    # Redacted messages (only present when decision='redact'). Always the TEXT projection: a message sent as content blocks comes back as a string. Attachments are never rewritten — we do not re-encode a PDF with the secret removed, and returning one that looked redacted would be worse than returning none. A message listed in `unscanned` as `over_size_budget` was not read, so it comes back as its text, unredacted.
    redacted_messages: list[GuardMessage] | Any
    # Detailed threat breakdown
    threats: list[ThreatDetail]
    # True when a finding flagged: allowed, reported, not enforced. The flagged findings are in `threats` with `flagged: true`, each with its threat type and rule. `decision` stays `allow` for a flag that decided, so a client that does not read this field lets the traffic through as before; on a `block` or `redact` it says a flag was also found. The event is stored with the decision `flag`.
    flagged: bool
    # On a `block`, how many independent detectors found the threat that decided it: 1 when one detector's verdict alone decided, 2 or more when others agreed. Counted by detector, not rule. 0 on anything but a `block`. A Shadow AI Device coaches rather than blocks a block resting on one detector (`basis` `detectors` and 1 here): the person may send it anyway, with a reason.
    detectors_agreeing: int
    # On a `block`, what it rests on: `secret` (a credential), `organisation_rule` (a policy, rule or pattern the organisation wrote), or `detectors` (the built-in detectors, as many as `detectors_agreeing` says). Null on anything but a `block`.
    basis: Literal["secret", "organisation_rule", "detectors"] | Any
    # Processing time in milliseconds
    latency_ms: float
    # Parts that reached us and were not read: they produced nothing to scan, or the scan budget was spent first. An `allow` with a non-empty `unscanned` is NOT 'this content is clean' — it is 'the text was clean and these parts were never read'. Reasons: url_only (we do not fetch caller-supplied URLs, that would be an SSRF primitive), file_id_unsupported, encrypted, no_text_extracted (a scanned/rasterised document), too_large, undecodable, unsupported_type, extractor_unavailable, unsupported_block, unsupported_tool_call (a tool call in none of the shapes we can read — `index` is its position among the request's tool calls), over_size_budget (a message older than the newest ones that filled the scan budget — `index` is its position in `messages`; it was not read on this call, and was the newest turn of an earlier one).
    unscanned: list[UnscannedAttachment]
    # Unavailable checks: detectors this scan would have run but the deployment cannot, because their backing service (an ML inference endpoint, an LLM judge, a bundled model) is not configured. An `allow` with a non-empty `unavailable` is NOT 'every check passed' — these detectors never looked. A detector that ran and found nothing is not listed, and neither is one the project's plan or guardrail settings leave out.
    unavailable: list[UnavailableCheck]
    # What this account is entitled to, at the instant this scan was answered. Null when the engine cannot say — an account with no subscription row, an admin key that bypasses the counter, or a row it could not read. Null is 'unknown', NOT 'unentitled': a client that reads it as a limit has invented a refusal the engine never made.
    entitlement: GuardEntitlement | Any


class GuardrailDelta(TypedDict, total=False):
    """Per-guardrail override the overlay wants to apply. Matches the ``guardrails`` override shape PolicyEngine already reads: ``{enabled, level, threshold}``. ``enabled=False`` disables a detector and is only ever a *loosening* op (surfaced as a critical warning)."""

    enabled: bool | Any
    level: Literal["strict", "moderate", "permissive"] | Any
    threshold: float | Any
    warn_threshold: float | Any
    block_threshold: float | Any


class GuardrailsConfig(TypedDict, total=False):
    """Full per-guardrail configuration for a project."""

    prompt_injection: PromptInjectionConfig
    pii_detection: PIIDetectionConfig
    toxicity: ToxicityConfig
    data_exfiltration: LevelConfig
    secret_key_detection: LevelConfig
    url_filtering: ToggleOnlyConfig
    fraud_detection: ToggleOnlyConfig
    malware_detection: ToggleOnlyConfig
    jailbreak_detection: ToggleOnlyConfig
    tool_injection: ToggleOnlyConfig
    multi_turn_drift: ToggleOnlyConfig
    hallucination: HallucinationConfig
    mcp_security: MCPSecurityConfig


class GuardrailsResponse(TypedDict):
    guardrails: GuardrailsConfig


class GuardrailsUpdateRequest(TypedDict):
    guardrails: GuardrailsConfig


class HTTPValidationError(TypedDict, total=False):
    detail: list[ValidationError]


class HallucinationConfig(TypedDict, total=False):
    enabled: bool
    action: Literal["metadata", "flag", "block"]
    block_threshold: float


class LevelConfig(TypedDict, total=False):
    """Guardrail with a strict/moderate/permissive sensitivity level."""

    enabled: bool
    level: Literal["strict", "moderate", "permissive"]


class MCPSecurityConfig(TypedDict, total=False):
    enabled: bool
    server_allowlist: list[str]
    server_blocklist: list[str]
    enforce_schema_validation: bool
    max_argument_size_bytes: int


class ManagedPolicyResponse(TypedDict, total=False):
    """The managed update policy an enrolled Shadow AI device should apply. ``fleet`` reflects the org's ``shadow_fleet_management`` entitlement; when false the other fields are null and the device keeps its local user preference."""

    fleet: bool
    force_update_mode: str | Any
    pinned_channel: str | Any
    min_version_override: str | Any


class ManifestFinding(TypedDict, total=False):
    # `E001` instructions injected into a description or parameter doc; `W021` invisible or look-alike characters; `E002` a name or reference that shadows another server's tool; `E003` an ATR community rule matched (advisory); `W022` two tools share a name, ignoring case; `E301` the list changed since `pin`; `E900` one tool could not be scanned, and was not.
    code: str
    # The tool's name, or `*` for the list as a whole
    tool: str
    # `high`, `medium` or `info`
    severity: str
    reason: str
    details: dict[str, Any]


class ManifestScanRequest(TypedDict, total=False):
    # The server's `tools/list` result: the `tools` array, sent as the server returned it. At most 500 tools, and 131072 characters of canonical JSON together with every name in `other_servers`.
    tools: list[McpTool]
    # The `sha256` an earlier scan of this server returned. Omit it the first time; when it differs from this list's `sha256`, the response carries a `E301` finding.
    pin: str | Any
    # The agent's other MCP servers, for shadowing: a tool here whose name one of them already has is a `E002` finding. Only their tool names are read, and they are not scanned themselves.
    other_servers: list[McpOtherServer]


class ManifestScanResponse(TypedDict):
    # This tool list's pin: send it as `pin` on the next scan of this server. Treat it as opaque.
    sha256: str
    # False when any finding's severity is `high`, or the scan is incomplete
    ok: bool
    findings: list[ManifestFinding]
    # Each tool's own hash, in the order sent, so a caller that keeps them can tell which tool changed. Not part of `sha256`. Opaque, like it.
    tools: list[ToolHash]
    # True when the scan ran out of its detection budget before its last tool. The tools it did not reach are in `unscanned_tools`; `sha256` and the rug-pull comparison are still complete.
    incomplete: bool
    # Names of the tools not scanned, in the order sent
    unscanned_tools: list[str]


class McpOtherServer(TypedDict):
    # Named in a shadowing finding
    name: str
    tools: list[McpToolName]


class McpTool(TypedDict, total=False):
    """One entry of an MCP ``tools/list`` result. Other keys are ignored."""

    name: str
    description: str | Any
    # The tool's JSON Schema for its arguments. `input_schema` is accepted too.
    inputSchema: dict[str, Any] | Any


class McpToolName(TypedDict):
    """A tool on another server: only its name is read. Other keys are ignored."""

    name: str


class MediaPartSchema(TypedDict, total=False):
    """A media attachment to be scanned for steganographic/adversarial payloads."""

    # Media type: 'image', 'audio' or 'document'
    type: str
    # MIME type, e.g. 'image/png', 'audio/wav'
    mime_type: str
    # URL to fetch the media from
    url: str | Any
    # Base64-encoded media data
    base64: str | Any
    # Extra metadata
    metadata: dict[str, Any] | Any


class OverlayApplyRequest(TypedDict, total=False):
    name: str
    delta: OverlayDelta
    project_id: str | Any
    acknowledge_loosening: bool


class OverlayDelta(TypedDict, total=False):
    """Additive deltas over a base policy config. Everything here is meant to *tighten*. Loosening ops (raising a threshold, dropping a detection level toward permissive, disabling a guardrail) are permitted to be expressed but are flagged as warnings in the diff."""

    detection_levels: dict[str, Any]
    toxicity_threshold: float | Any
    add_custom_patterns: list[str]
    add_blocked_domains: list[str]
    guardrails: dict[str, Any]


class OverlayOut(TypedDict):
    id: str
    name: str
    version: int
    status: str
    project_id: str | Any
    warnings: list[OverlayWarningOut]


class OverlayPreviewRequest(TypedDict, total=False):
    delta: OverlayDelta
    sample: SampleSource
    max_examples: int
    project_id: str | Any


class OverlayWarningOut(TypedDict):
    kind: str
    field: str
    message: str
    severity: Literal["warning", "critical"]


class PIIDetectionConfig(TypedDict, total=False):
    enabled: bool
    level: Literal["strict", "moderate", "permissive"]
    mode: Literal["redact", "mask", "block", "tokenize"]
    entities: list[str] | Any


class ProjectResponse(TypedDict, total=False):
    id: str
    name: str
    description: str | Any
    fail_mode: str
    use_case: str
    strictness_level: str
    zero_retention: bool
    created_at: str


class PromptInjectionConfig(TypedDict, total=False):
    """The prompt-injection guardrail: a level, and optionally bands."""

    # A score at or above this and below `block_threshold` flags: the request is allowed and the finding is reported. Needs `block_threshold`.
    warn_threshold: float | Any
    # A score at or above this blocks. Null: the detector decides as before.
    block_threshold: float | Any
    enabled: bool
    level: Literal["strict", "moderate", "permissive"]


class ProxyAnthropicMessage(TypedDict):
    """An Anthropic message."""

    id: str
    type: str
    role: str
    model: str
    content: list[dict[str, Any]]


class ProxyChatCompletion(TypedDict):
    """An OpenAI chat completion."""

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]


class ProxyModelList(TypedDict):
    """OpenAI's model list."""

    data: list[dict[str, Any]]


class ProxyResponsesObject(TypedDict):
    """An OpenAI Responses API response."""

    id: str
    object: str
    model: str
    output: list[dict[str, Any]]


class ProxyTokenCount(TypedDict):
    """Anthropic's token count for a messages body."""

    input_tokens: int


class QuotaErrorDetail(TypedDict, total=False):
    message: str
    # 'quota_exceeded' or 'spending_limit_exceeded'
    type: str
    # 'monthly_quota_exceeded' or 'spending_limit_exceeded'
    code: str
    current_plan: str
    requests_used: int
    requests_limit: int
    upgrade_url: str
    retry_after: int | Any


class QuotaErrorEnvelope(TypedDict):
    error: QuotaErrorDetail


class RedactRequest(TypedDict, total=False):
    # Text to redact
    content: str
    # Entity types to redact. Omit to use the policy's configured entities. Accepts detector entity names ('email', 'ssn', 'credit_card', 'phone_us'), the family aliases 'phone', 'ip_address' and 'passport', and 'api_key'. An unrecognized name is rejected rather than ignored.
    pii_types: list[str] | Any


class RedactResponse(TypedDict):
    original: str
    redacted: str
    piiFound: list[str]


class SampleSource(TypedDict, total=False):
    """Where the shadow traffic sample comes from. ``corpus`` reuses the curated Shadow eval corpus (offline, deterministic); ``inline`` lets the caller pass their own recent-traffic prompts."""

    kind: Literal["corpus", "inline"]
    limit: int
    texts: list[str]


class ScanRequest(TypedDict, total=False):
    # Text to scan
    content: str
    # Content type: 'prompt' or 'response'
    type: str


class ScanResponse(TypedDict, total=False):
    blocked: bool
    decision: str
    # True when a detector's warn threshold flagged the content (#970): it was allowed, not blocked, and the event is recorded as flagged. Read `threshold` for why. A client must not report a flagged result as clean.
    flagged: bool
    # The warn or block threshold that decided, when one did. Null otherwise.
    threshold: AppliedThreshold | Any
    reason: str
    threatType: str | Any
    confidence: float
    eventId: str
    processingTimeMs: float


class ShadowReportOut(TypedDict):
    total: int
    counts: dict[str, Any]
    blocked_base: int
    blocked_candidate: int
    warnings: list[OverlayWarningOut]
    examples: dict[str, Any]


class SiemEventRecord(TypedDict, total=False):
    """One security event, in the SIEM event contract (schema_version 1)."""

    schema_version: str
    # `threat_detected` or `scan_completed`.
    event: str
    event_id: str
    # When the event happened, ISO 8601 with offset.
    timestamp: str
    organization_id: str | Any
    organization_name: str | Any
    project_id: str | Any
    # The project's name.
    project: str | Any
    # Null for a key no Device holds.
    device_id: str | Any
    device_hostname: str | Any
    # Free text naming who is believed to be behind the Device. Never validated; read `identity_verified` before treating it as a person.
    attribution_label_unverified: str | Any
    # True when a Member signed in to bind the Device; false when only the label stands behind it; null when the event predates the record.
    identity_verified: bool | Any
    threat_type: str | Any
    decision: str
    confidence: float | Any
    reason: str | Any
    direction: str | Any
    surface: str | Any
    # On a Shadow AI Device's event, the part of the request the finding came from: `current_turn`, `earlier_turn`, `instructions` or `tool_definition`. Null for any other caller's.
    segment: str | Any
    # Opens this event in its Organization.
    dashboard_url: str
    text: str
    threshold: dict[str, Any] | Any


class TestCatalog(TypedDict):
    """What a caller can run -- so a CLI can list before it picks."""

    total: int
    tests: list[TestInfo]


class TestInfo(TypedDict):
    """One attack in the corpus, without running it."""

    name: str
    category: str
    description: str
    expected_result: str


class TestRequest(TypedDict, total=False):
    """Body for run-all (where custom_prompt is ignored) and run-custom. ``target_preset`` names the policy to test. A composed ``"use_case:strictness"`` pair is used as given; a bare use case means that use case at moderate strictness; a bare strictness level means the default use case at that strictness -- so the wire default ``"default"`` is ``default:moderate`` and ``"strict"`` is ``default:strict``. The vocabularies are the ones served by ``GET /dashboard/presets/use-cases`` and ``GET /dashboard/presets/strictness-levels``. Anything else is a 400 that lists the valid composed names; it used to be silently substituted with the default engine, which made every preset selector a no-op."""

    custom_prompt: str | Any
    # Policy to test: 'use_case:strictness', a bare use case (moderate strictness), or a bare strictness level (default use case). Unknown names are rejected with 400 invalid_preset.
    target_preset: str


class TestResponse(TypedDict, total=False):
    """One attack prompt and what the policy engine decided about it."""

    test_name: str
    prompt: str
    # "block", "allow", or "error" when the probe did not complete.
    decision: str
    reason: str
    threat_type: str | Any
    confidence: float | Any
    blocked: bool
    # True when the engine raised while evaluating this prompt. Not a verdict: an errored probe is neither blocked nor allowed, and is excluded from the summary's block_rate.
    errored: bool


class TestSummary(TypedDict, total=False):
    """The whole corpus, plus the block rate that is the headline number."""

    total_tests: int
    blocked: int
    allowed: int
    # blocked / total_tests over COMPLETED probes; 0.0 for an empty corpus, never a divide-by-zero. Errored probes are excluded from total_tests.
    block_rate: float
    # Probes the engine could not complete (the evaluator raised). Excluded from total_tests and the block rate; a CI caller can fail on this to catch 'the scan did not complete' rather than mistaking it for a pass.
    errored_count: int
    # Which arithmetic produced block_rate. 2: errored probes are reported in errored_count and excluded from the rate. 1 (never sent; before 2026-09-03) graded evaluator exceptions by their message text.
    report_version: int
    results: list[TestResponse]


class ThreatDetail(TypedDict, total=False):
    """Individual threat found during scanning."""

    type: str
    confidence: float
    details: str
    # severity_score * confidence, clamped to [0, 1]. The decision-driving number when a severity-carrying detector (e.g. structural heuristics) fired; null when confidence alone is the signal.
    weighted_score: float | Any
    # The attack technique an injection or jailbreak finding belongs to, as ATR names it: a dotted category and subcategory such as `prompt-injection.persona-hijacking`. When several findings disagree, the highest-scoring one names it. Null for a finding that is not an attack, such as redacted PII.
    technique: str | Any
    # The detector and rule that made this finding, the pair a Suppression is keyed on. On a block, the rule that decided.
    rule: FindingRule | Any
    # The part of the request the finding came from. On a Shadow AI Device's scan: `current_turn` (the person's new message, its tool results and the tool calls they answer), `earlier_turn` (a turn already sent, redacted again because the app resends it), or `instructions` (the app's system or developer prompt, scanned for secrets only). On any caller's scan: `tool_definition`, for an entry of `context.tool_definitions`. Null otherwise.
    segment: str | Any
    # Whether this finding decides. `active`: it does, and is what `decision`, `confidence` and `threat_type` report. `candidate`: a Candidate rule matched. It is recorded so the rule can be measured on live traffic, and it never changes the decision, the score or the threat type; a `candidate` threat can appear on an `allow`. `suppressed`: one of the project's Suppressions silenced the rule, so it is recorded and decides nothing, and another rule may decide instead. Decide on `decision`, never on whether `threats` is empty.
    status: FindingStatus
    # The warn and block thresholds this finding's score was compared with, when the project set them for its detector (#970): which band it fell in, both thresholds and the score. A `warn` band finding appears on an `allow`: it was reported, not enforced. Null when the detector has no thresholds configured and decided as it always has, and on a finding that did not decide.
    threshold: AppliedThreshold | Any
    # True when this finding flagged: it was reported and not enforced. A warn band raised it (`threshold` says so), a custom policy rule or LLM guard whose action is `flag`, an agentic judge's `flag`, or a detector in `flag` mode. A flagged finding can sit beside the `block` or `redact` that decided; it never changes `decision`.
    flagged: bool
    # Lowercase hex SHA-256 of the UTF-8 text this finding's rule matched: the `match_sha256` a Suppression names to silence the rule for that text only. Null when the rule cannot be suppressed, or the finding has no one matched text (a redaction by type).
    match_sha256: str | Any


# Types of threats detected. Originals (block 1) cover prompt injection, PII, toxicity, exfiltration, fraud, and malicious tool/MCP invocation. Block 2 (added 2026-04) covers the AI Agent Traps framework (Franklin et al., Google DeepMind 2025) -- six categories of environment-driven attacks against autonomous agents. Cross-link to the per-category remediation pages under apps/docs/security/.
ThreatType = Literal[
    "prompt_injection",
    "pii_leak",
    "data_exfiltration",
    "toxicity",
    "api_key_leak",
    "system_prompt_leak",
    "policy_violation",
    "fraud_abuse",
    "malware",
    "secret_key_leak",
    "url_violation",
    "malicious_entity",
    "off_topic",
    "mcp_violation",
    "insecure_code",
    "gibberish",
    "language_violation",
    "multi_turn_escalation",
    "html_obfuscation",
    "syntactic_masking",
    "image_stego",
    "image_adversarial",
    "audio_stego",
    "font_injection",
    "dynamic_cloaking",
    "framing_bias",
    "critic_evasion",
    "persona_hyperstition",
    "rag_poisoning",
    "memory_poisoning",
    "few_shot_poisoning",
    "sub_agent_spawning",
    "compositional_fragment",
    "sybil_attack",
    "systemic_cascade",
    "tacit_collusion",
    "approval_fatigue",
]


class ToggleOnlyConfig(TypedDict, total=False):
    enabled: bool


class ToolHash(TypedDict):
    name: str
    sha256: str


class ToxicityConfig(TypedDict, total=False):
    # A score at or above this and below `block_threshold` flags: the request is allowed and the finding is reported. Needs `block_threshold`.
    warn_threshold: float | Any
    # A score at or above this blocks. Null: the detector decides as before.
    block_threshold: float | Any
    enabled: bool
    threshold: float
    categories: list[str] | Any


class UnavailableCheck(TypedDict):
    """One detector the deployment cannot run, and why."""

    # Stable detector name, e.g. 'jailbreak_judge'.
    detector: str
    # Why it cannot run: ml_inference_off (ML_INFERENCE_MODE=off), ml_inference_not_configured (no inference endpoint or token), ml_inference_mode_unsupported (a mode this detector does not implement), llm_judge_not_configured (no LLM judge endpoint), model_not_bundled (an in-process model artifact is missing).
    reason: str


class UnscannedAttachment(TypedDict):
    """One part of the request we could not read, and why."""

    # Position within the list the reason names — the combined attachment list; the request's tool calls for `unsupported_tool_call` (`context.tool_calls` first, then each message's in order); `messages` for `over_size_budget`. -1 when the part has no position, which is every `unsupported_block`.
    index: int
    # Stable machine-readable code
    reason: str
    # Reason with any extra qualifier, e.g. 'no_text_extracted:pages=3'
    detail: str


class UsageStatsResponse(TypedDict, total=False):
    """The caller's request usage this month. ``reset_date`` is ISO 8601."""

    has_subscription: bool
    plan: str
    status: str | Any
    monthly_limit: int
    used_this_month: int
    remaining: int
    usage_percentage: float
    reset_date: str | Any


class ValidationError(TypedDict, total=False):
    loc: list[str | int]
    msg: str
    type: str
    input: Any
    ctx: dict[str, Any]


class WebhookAckResponse(TypedDict):
    """What GitHub is told: a ping is answered, every other event accepted for later."""

    status: Literal["pong", "accepted"]


class developer__agent__router__ToolCallRequest(TypedDict, total=False):
    """Request to validate a tool call"""

    agent_id: str
    tool_name: str
    arguments: dict[str, Any]
    session_id: str | Any


class developer__agent__router__ToolCallResponse(TypedDict, total=False):
    """Response from tool call validation"""

    allowed: bool
    risk_score: float
    risk_level: str
    reason: str
    warnings: list[str]
    blocked_reasons: list[str]


class developer__policies__router__AgentPolicy(TypedDict, total=False):
    """A single enforced rule, flattened for the agent UI."""

    id: str
    name: str
    description: str | Any
    action: str
    threat_types: list[str]
    priority: int


class developer__projects__schemas__CreateProjectRequest(TypedDict, total=False):
    name: str
    description: str | Any
    # Behaviour when the detection engine errors: 'open' forwards the request, 'closed' rejects it with 503.
    fail_mode: Literal["open", "closed"]
    use_case: str
    strictness_level: Literal["strict", "moderate", "permissive"]
