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


class ApiKeyFullResponse(TypedDict):
    """Response containing the full API key for copy functionality"""

    id: str
    name: str
    prefix: str
    key: str


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


class GuardContext(TypedDict, total=False):
    """Optional rich context from framework integrations. Only ``tool_calls`` is scanned. The rest is descriptive — it labels the event for the dashboard and the audit log, and does not reach a detector."""

    # Framework name, e.g. 'langchain', 'crewai'
    framework: str | Any
    # LangChain chain name or pipeline identifier
    chain_name: str | Any
    # Agent identifier for multi-agent systems
    agent_id: str | Any
    # Session identifier for multi-turn tracking
    session_id: str | Any
    # Tool calls in this turn. The tool NAME and its ARGUMENTS are assembled into the scanned text and get the same detection stack as the messages — tool arguments are where an exfiltration payload actually travels, so they are scanned rather than logged. Both provider spellings are read: OpenAI's `{'type':'function','function':{'name','arguments'}}` and Anthropic's `{'type':'tool_use','name','input'}`, plus LangChain's `{'name','args'}`. A call in none of those shapes is reported in the response's `unscanned` array with its position; it is never quietly skipped.
    tool_calls: list[dict[str, Any]] | Any
    # Arbitrary framework-specific metadata (not scanned)
    metadata: dict[str, Any] | Any


class GuardMessage(TypedDict, total=False):
    """A single message in the conversation. ``content`` takes either a plain string or a provider-shaped content-block array — OpenAI's ``text``/``image_url``/``input_audio``/``file`` and Anthropic's ``text``/``image``/``document`` are all understood, because those are the two shapes our own proxy already receives. Blocks are accepted as loose dicts rather than a closed union on purpose. Both providers add block types faster than we can model them, and a strict schema would 422 a request we could otherwise have scanned the text of. Anything unrecognised is *reported* in the response's ``unscanned`` rather than dropped — see ``shared.security.content_parts``."""

    # Message role: system, user, assistant, tool
    role: str
    # Message text, or an OpenAI/Anthropic content-block array. Attachments carried in blocks are extracted and scanned like any other text; blocks we cannot read are listed in `unscanned`.
    content: str | list[dict[str, Any]]


class GuardRequest(TypedDict, total=False):
    """Request body for the guard endpoint."""

    # Messages to scan (OpenAI-style message array)
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
    # Redacted messages (only present when decision='redact'). Always the TEXT projection: a message sent as content blocks comes back as a string. Attachments are never rewritten — we do not re-encode a PDF with the secret removed, and returning one that looked redacted would be worse than returning none.
    redacted_messages: list[GuardMessage] | Any
    # Detailed threat breakdown
    threats: list[ThreatDetail]
    # Processing time in milliseconds
    latency_ms: float
    # Parts that reached us and produced nothing to scan. An `allow` with a non-empty `unscanned` is NOT 'this content is clean' — it is 'the text was clean and these parts were never read'. Reasons: url_only (we do not fetch caller-supplied URLs, that would be an SSRF primitive), file_id_unsupported, encrypted, no_text_extracted (a scanned/rasterised document), too_large, undecodable, unsupported_type, extractor_unavailable, unsupported_block, unsupported_tool_call (an entry in `context.tool_calls` in none of the shapes we can read — `index` is its position in that list).
    unscanned: list[UnscannedAttachment]


class GuardrailDelta(TypedDict, total=False):
    """Per-guardrail override the overlay wants to apply. Matches the ``guardrails`` override shape PolicyEngine already reads: ``{enabled, level, threshold}``. ``enabled=False`` disables a detector and is only ever a *loosening* op (surfaced as a critical warning)."""

    enabled: bool | Any
    level: Literal["strict", "moderate", "permissive"] | Any
    threshold: float | Any


class GuardrailsConfig(TypedDict, total=False):
    """Full per-guardrail configuration for a project."""

    prompt_injection: LevelConfig
    pii_detection: PIIDetectionConfig
    toxicity: ToxicityConfig
    data_exfiltration: LevelConfig
    secret_key_detection: LevelConfig
    url_filtering: ToggleOnlyConfig
    fraud_detection: ToggleOnlyConfig
    malware_detection: ToggleOnlyConfig
    jailbreak_detection: ToggleOnlyConfig
    tool_injection: ToggleOnlyConfig
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
    """The managed update policy an enrolled Shadow AI device should apply. ``fleet`` reflects the org's ``shadow_ai_fleet`` entitlement; when false the other fields are null and the device keeps its local user preference."""

    fleet: bool
    force_update_mode: str | Any
    pinned_channel: str | Any
    min_version_override: str | Any


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
    mode: Literal["redact", "mask", "block"]
    entities: list[str] | Any


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
    """Body for run-all (where custom_prompt is ignored) and run-custom."""

    custom_prompt: str | Any
    target_preset: str


class TestResponse(TypedDict):
    """One attack prompt and what the policy engine decided about it."""

    test_name: str
    prompt: str
    decision: str
    reason: str
    threat_type: str | Any
    confidence: float | Any
    blocked: bool


class TestSummary(TypedDict):
    """The whole corpus, plus the block rate that is the headline number."""

    total_tests: int
    blocked: int
    allowed: int
    # blocked / total_tests; 0.0 for an empty corpus, never a divide-by-zero.
    block_rate: float
    results: list[TestResponse]


class ThreatDetail(TypedDict, total=False):
    """Individual threat found during scanning."""

    type: str
    confidence: float
    details: str
    # severity_score * confidence, clamped to [0, 1]. The decision-driving number when a severity-carrying detector (e.g. structural heuristics) fired; null when confidence alone is the signal.
    weighted_score: float | Any


class ToggleOnlyConfig(TypedDict, total=False):
    enabled: bool


class ToxicityConfig(TypedDict, total=False):
    enabled: bool
    threshold: float
    categories: list[str] | Any


class UnscannedAttachment(TypedDict):
    """One part of the request we could not read, and why."""

    # Position within the list the reason names — the combined attachment list, or `context.tool_calls` for `unsupported_tool_call`. -1 when the part has no position, which is every `unsupported_block`.
    index: int
    # Stable machine-readable code
    reason: str
    # Reason with any extra qualifier, e.g. 'no_text_extracted:pages=3'
    detail: str


class ValidationError(TypedDict, total=False):
    loc: list[str | int]
    msg: str
    type: str
    input: Any
    ctx: dict[str, Any]


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


class developer__projects__schemas__ProjectResponse(TypedDict, total=False):
    id: str
    name: str
    description: str | Any
    fail_mode: str
    use_case: str
    strictness_level: str
    zero_retention: bool
    created_at: str
