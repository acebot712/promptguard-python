"""
Auto-generated from OpenAPI spec (v1.0.0).
DO NOT EDIT — regenerate with: python scripts/generate_types_from_spec.py

These are type-only definitions. Custom client logic lives in
promptguard/guard.py, promptguard/client.py, promptguard/patches/,
and promptguard/integrations/ — those files are never modified
by this generator.
"""

from __future__ import annotations

from typing import Any, TypedDict

"""Request to register a new agent identity."""

class AgentRegisterRequest(TypedDict, total=False):
    agent_name: str
    allowed_tools: list[str] | Any


"""Response from agent registration — secret is shown only once."""

class AgentRegisterResponse(TypedDict):
    agent_id: str
    agent_name: str
    agent_secret: str
    credential_prefix: str


"""Response from credential rotation."""

class AgentRotateResponse(TypedDict):
    agent_id: str
    new_secret: str
    credential_prefix: str
    old_credential_revoked: bool


"""Statistics for an agent"""

class AgentStats(TypedDict):
    agent_id: str
    total_tool_calls: int
    blocked_calls: int
    avg_risk_score: float
    active_sessions: int
    anomalies_detected: int


"""Response containing the full API key for copy functionality"""

class ApiKeyFullResponse(TypedDict):
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


"""Request to run the autonomous red team agent."""

class AutonomousRequest(TypedDict, total=False):
    budget: int
    target_preset: str
    enabled_detectors: list[str] | Any


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


class ErrorDetail(TypedDict):
    # Human-readable error description
    message: str
    # Error category, e.g. 'authentication_error'
    type: str
    # Machine-readable error code
    code: str


"""Optional rich context from framework integrations."""

class GuardContext(TypedDict, total=False):
    # Framework name, e.g. 'langchain', 'crewai'
    framework: str | Any
    # LangChain chain name or pipeline identifier
    chain_name: str | Any
    # Agent identifier for multi-agent systems
    agent_id: str | Any
    # Session identifier for multi-turn tracking
    session_id: str | Any
    # Tool calls in this turn
    tool_calls: list[dict[str, Any]] | Any
    # Arbitrary framework-specific metadata
    metadata: dict[str, Any] | Any


"""A single message in the conversation."""

class GuardMessage(TypedDict, total=False):
    # Message role: system, user, assistant, tool
    role: str
    # Message text content
    content: str


"""Request body for the guard endpoint."""

class GuardRequest(TypedDict, total=False):
    # Messages to scan (OpenAI-style message array)
    messages: list[GuardMessage]
    # Scan direction: 'input' (pre-LLM) or 'output' (post-LLM)
    direction: str
    # Model being used (for logging)
    model: str | Any
    # Optional framework context
    context: GuardContext | Any


"""Response from the guard endpoint."""

class GuardResponse(TypedDict, total=False):
    # Policy decision: 'allow', 'block', or 'redact'
    decision: str
    # Unique event identifier for tracking
    event_id: str
    # Confidence score of the decision
    confidence: float
    # Primary threat type detected
    threat_type: str | Any
    # Redacted messages (only present when decision='redact')
    redacted_messages: list[GuardMessage] | Any
    # Detailed threat breakdown
    threats: list[ThreatDetail]
    # Processing time in milliseconds
    latency_ms: float


class HTTPValidationError(TypedDict, total=False):
    detail: list[ValidationError]


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
    # Specific PII types to redact (default: all)
    pii_types: list[str] | Any


class RedactResponse(TypedDict):
    original: str
    redacted: str
    piiFound: list[str]


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


"""Individual threat found during scanning."""

class ThreatDetail(TypedDict):
    type: str
    confidence: float
    details: str


class ValidationError(TypedDict, total=False):
    loc: list[str | int]
    msg: str
    type: str
    input: Any
    ctx: dict[str, Any]


"""Request to validate a tool call"""

class developer__agent__router__ToolCallRequest(TypedDict, total=False):
    agent_id: str
    tool_name: str
    arguments: dict[str, Any]
    session_id: str | Any


"""Response from tool call validation"""

class developer__agent__router__ToolCallResponse(TypedDict, total=False):
    allowed: bool
    risk_score: float
    risk_level: str
    reason: str
    warnings: list[str]
    blocked_reasons: list[str]


class developer__projects__schemas__CreateProjectRequest(TypedDict, total=False):
    name: str
    description: str | Any
    fail_mode: str
    use_case: str
    strictness_level: str


class developer__projects__schemas__ProjectResponse(TypedDict, total=False):
    id: str
    name: str
    description: str | Any
    fail_mode: str
    use_case: str
    strictness_level: str
    zero_retention: bool
    created_at: str


"""Request to run a red team test"""

class internal__redteam__router__TestRequest(TypedDict, total=False):
    custom_prompt: str | Any
    target_preset: str


"""Response from a red team test"""

class internal__redteam__router__TestResponse(TypedDict):
    test_name: str
    prompt: str
    decision: str
    reason: str
    threat_type: str | Any
    confidence: float
    blocked: bool
    details: dict[str, Any]


"""Summary of all red team tests"""

class internal__redteam__router__TestSummary(TypedDict):
    total_tests: int
    blocked: int
    allowed: int
    block_rate: float
    results: list[internal__redteam__router__TestResponse]

