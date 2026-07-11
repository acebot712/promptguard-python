"""PromptGuard SDK Configuration"""

from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for PromptGuard SDK"""

    # ``repr=False`` keeps the API key out of ``repr(config)`` / logs / tracebacks.
    api_key: str = field(repr=False)
    base_url: str = "https://api.promptguard.co/api/v1/proxy"
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0

    def __post_init__(self) -> None:
        # Clamp so a hand-built ``Config`` can never collapse the request loop
        # to zero attempts or feed httpx a negative timeout/delay.
        self.max_retries = max(0, self.max_retries)
        self.retry_delay = max(0.0, self.retry_delay)
        self.timeout = max(0.0, self.timeout)

    def __repr__(self) -> str:
        key = self.api_key or ""
        masked = f"{key[:6]}…{key[-2:]}" if len(key) > 8 else "***"
        return (
            f"Config(api_key='{masked}', base_url={self.base_url!r}, "
            f"max_retries={self.max_retries}, retry_delay={self.retry_delay}, "
            f"timeout={self.timeout})"
        )
