"""Configuration models using Pydantic.

LEARNING NOTES:
- Pydantic models validate data automatically at construction time.
- `extra="forbid"` rejects unknown fields — catches typos in config files.
- `field_validator` lets you add custom validation logic per field.
- `model_validator` validates across multiple fields together.
- Compare with LCS: src/models/config.py uses the same patterns.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing_extensions import Self


class ConfigurationBase(BaseModel):
    """Base configuration class that rejects unknown fields.

    In the real LCS, this prevents misconfiguration — if you typo a field
    name in your YAML config, you get an error instead of silent ignoring.
    """

    model_config = {"extra": "forbid"}


class LlamaStackConfig(ConfigurationBase):
    """Configuration for connecting to a Llama Stack server.

    LEARNING: Llama Stack is Meta's open-source framework for building
    AI applications. It provides a standard API for inference, agents,
    RAG, and tool use. The Lightspeed Stack uses it as its AI engine.
    """

    url: str = Field(
        default="http://localhost:5001",
        description="URL of the Llama Stack server",
    )
    model: str = Field(
        default="meta-llama/Llama-3.2-3B-Instruct",
        description="Model identifier to use for inference",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        description="Request timeout in seconds",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure URL starts with http:// or https://."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v.rstrip("/")


class KnowledgeBaseConfig(ConfigurationBase):
    """Configuration for the knowledge base (simplified RAG).

    LEARNING: In the real LCS, RAG (Retrieval-Augmented Generation) uses
    vector databases like FAISS or Pinecone to store document embeddings.
    When a user asks a question, relevant docs are retrieved and injected
    into the LLM prompt as context. This is how the LLM "knows" about
    OpenShift-specific topics.
    """

    enabled: bool = Field(default=True, description="Enable knowledge base lookups")
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include a knowledge chunk",
    )


class CORSConfig(ConfigurationBase):
    """CORS middleware configuration."""

    allow_origins: list[str] = Field(default=["*"])
    allow_methods: list[str] = Field(default=["*"])
    allow_headers: list[str] = Field(default=["*"])


class AuthConfig(ConfigurationBase):
    """Authentication configuration.

    LEARNING: The real LCS supports multiple auth backends:
    - Kubernetes ServiceAccount tokens
    - JWK (JSON Web Key) based auth
    - Red Hat Identity headers
    - NoOp (for development)
    Our simplified version just uses API key auth.
    """

    enabled: bool = Field(default=False, description="Enable API key authentication")
    api_key: str = Field(default="dev-key-12345", description="API key for authentication")


class AppConfig(ConfigurationBase):
    """Top-level application configuration.

    LEARNING: In LCS, this is loaded from a YAML file specified by
    the LIGHTSPEED_STACK_CONFIG_PATH environment variable. We do the same.
    """

    name: str = Field(default="OpenShift Chat Assistant")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, ge=1, le=65535)
    llama_stack: LlamaStackConfig = Field(default_factory=LlamaStackConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    @model_validator(mode="after")
    def warn_no_auth(self) -> Self:
        """Log a warning when auth is disabled.

        LEARNING: model_validator(mode='after') runs after all fields
        are validated. This lets you validate relationships between fields.
        Compare with LCS src/models/config.py model validators.
        """
        if not self.auth.enabled:
            import logging

            logging.getLogger(__name__).warning(
                "Authentication is disabled — not suitable for production!"
            )
        return self
