"""PromptGuard SDK Configuration"""

from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for PromptGuard SDK"""

    api_key: str
    base_url: str = "https://api.promptguard.co/api/v1/proxy"
    max_retries: int = 3
    retry_delay: float = 1.0
