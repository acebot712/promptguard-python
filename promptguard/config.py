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

    # Timeouts (seconds)
    timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
