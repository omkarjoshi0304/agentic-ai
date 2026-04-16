"""FastAPI application — the main entry point.

LEARNING NOTES:
- This mirrors LCS's src/app/main.py structure:
  1. Load configuration from YAML
  2. Initialize services in the lifespan context
  3. Register routers (endpoints)
  4. Add middleware (CORS, auth, metrics)
- FastAPI's lifespan replaces the old @app.on_event("startup") pattern.
- Compare with LCS: src/app/main.py

KEY CONCEPTS:
- Lifespan: async context manager for startup/shutdown logic
- Routers: modular endpoint groups (like Express routers or Flask blueprints)
- Middleware: request/response processing pipeline (auth, CORS, metrics)
- Dependency Injection: FastAPI's Depends() for clean, testable code
"""

import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models.config import AppConfig
from services.conversation import ConversationStore
from services.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Global app state — populated during lifespan startup
# LEARNING: In LCS, these are singletons (Configuration, AsyncLlamaStackClientHolder).
# We use a simple dict for clarity.
app_state: dict = {}


def load_config() -> AppConfig:
    """Load configuration from YAML file or use defaults.

    LEARNING: In LCS, config is loaded from the path specified by
    LIGHTSPEED_STACK_CONFIG_PATH env var. We do the same, but also
    support running with defaults for easy experimentation.
    """
    config_path = os.environ.get("APP_CONFIG_PATH")

    if config_path and os.path.exists(config_path):
        logger.info("Loading configuration from %s", config_path)
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        return AppConfig(**raw_config)

    logger.info("No config file found — using defaults")
    return AppConfig()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up application resources.

    LEARNING: The lifespan context manager is FastAPI's way of handling
    startup and shutdown. Everything before `yield` runs at startup;
    everything after runs at shutdown.

    Compare with LCS's lifespan in src/app/main.py:
    - Loads config from YAML
    - Initializes Llama Stack client
    - Registers MCP servers
    - Sets up database
    """
    # --- STARTUP ---
    logger.info("Starting OpenShift Chat Assistant")

    # Load configuration (Pydantic validates it automatically)
    config = load_config()
    app_state["config"] = config
    logger.info("Configuration loaded: %s", config.name)

    # Initialize LLM client (connects to Llama Stack)
    llm_client = LLMClient(config.llama_stack)
    await llm_client.check_connection()
    app_state["llm_client"] = llm_client

    # Initialize conversation store
    app_state["conversation_store"] = ConversationStore()

    logger.info("App startup complete — ready to serve requests")

    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down...")
    await llm_client.close()
    logger.info("Shutdown complete")


# Create the FastAPI app
app = FastAPI(
    title="OpenShift Chat Assistant",
    summary="AI-powered chat assistant for OpenShift — a learning project",
    description=(
        "A simplified version of the Lightspeed Stack architecture. "
        "This project teaches FastAPI, Pydantic, and Llama Stack concepts "
        "by building an OpenShift troubleshooting assistant."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# --- MIDDLEWARE ---
# LEARNING: Middleware processes every request/response in a pipeline.
# Order matters — last added middleware runs first (outermost).
# Compare with LCS: CORSMiddleware, RestApiMetricsMiddleware, GlobalExceptionMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be overridden by config in lifespan
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTERS ---
# LEARNING: Routers group related endpoints. In LCS, routers.include_routers()
# adds all endpoint modules. We do it explicitly here for clarity.

from app.endpoints import chat, health, knowledge

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(knowledge.router)


# --- ROOT ENDPOINT ---
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — service info.

    LEARNING: In LCS, this is in src/app/endpoints/root.py.
    It's useful for quick "is it running?" checks.
    """
    return {
        "service": "OpenShift Chat Assistant",
        "version": "0.1.0",
        "docs": "/docs",
        "description": (
            "A learning project inspired by the Lightspeed Stack. "
            "Visit /docs for the interactive API documentation."
        ),
    }
