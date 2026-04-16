"""Authentication middleware.

LEARNING NOTES:
- FastAPI uses a dependency injection system for auth — clean and testable.
- The real LCS supports multiple auth backends via a factory pattern:
  - Kubernetes ServiceAccount token validation
  - JWK (JSON Web Key) for external identity providers
  - Red Hat Identity header-based auth
  - NoOp for development
- Our simplified version uses API key auth to teach the concept.
- Compare with LCS: src/authentication/
"""

import logging

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from models.config import AuthConfig

logger = logging.getLogger(__name__)

# FastAPI's APIKeyHeader automatically extracts the header from requests
# and handles missing header errors with a 403 response.
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_auth_dependency(auth_config: AuthConfig):
    """Create an auth dependency based on configuration.

    LEARNING: This is a factory function that returns a FastAPI dependency.
    FastAPI's Depends() system injects the return value into endpoint
    functions. This pattern lets you swap auth backends without changing
    endpoint code — exactly how LCS does it.

    In LCS, `get_auth_dependency()` in src/authentication/__init__.py
    returns the right authenticator based on config.
    """

    async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
        """Verify the API key and return the user identifier.

        LEARNING: In the real LCS, this would validate a Kubernetes
        ServiceAccount token and extract the user's identity and RBAC
        permissions. The returned user info is used for quota tracking
        and audit logging.
        """
        if not auth_config.enabled:
            return "anonymous"

        if api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide X-API-Key header.",
            )

        if api_key != auth_config.api_key:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key.",
            )

        logger.info("Authenticated request with valid API key")
        return "authenticated-user"

    return verify_api_key
