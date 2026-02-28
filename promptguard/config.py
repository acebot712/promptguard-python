"""
PromptGuard SDK Configuration
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for PromptGuard SDK"""

    # Authentication
    api_key: str

    # API endpoint
    base_url: str = "https://api.promptguard.co/api/v1/proxy"

    # Features
    enable_caching: bool = True
    enable_security_scan: bool = True
    enable_memory: bool = True

    # Timeouts (seconds)
    timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Project settings
    project_id: str | None = None

    # Debug
    debug: bool = False
