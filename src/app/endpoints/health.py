"""Health check endpoint.

LEARNING NOTES:
- Every production service needs health checks for Kubernetes liveness/readiness probes.
- In OpenShift, the kubelet calls these endpoints to decide whether to restart a pod.
- Compare with LCS: src/app/endpoints/health.py
"""

from fastapi import APIRouter

from models.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/healthz",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and Llama Stack is connected.",
)
async def health_check() -> HealthResponse:
    """Return service health status.

    LEARNING: In OpenShift, you'd configure this as a liveness probe:
      livenessProbe:
        httpGet:
          path: /healthz
          port: 8080
        initialDelaySeconds: 5
        periodSeconds: 10
    """
    # Access the app state (set during lifespan startup)
    from app.main import app_state

    return HealthResponse(
        status="healthy",
        service_name=app_state["config"].name,
        llama_stack_connected=app_state["llm_client"].is_connected,
    )
