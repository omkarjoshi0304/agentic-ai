# OpenShift Chat Assistant

A learning project that teaches **FastAPI**, **Pydantic**, **Llama Stack**, and the **Lightspeed Stack architecture** by building a simplified AI chat assistant for OpenShift.

## Architecture

This project implements **Option C (Hybrid)** from the Lightspeed Stack architecture:

```
┌─────────────────────────────────────────────────────┐
│              This Project (Server-Side)              │
│                                                      │
│  FastAPI App ──▶ Knowledge Base (RAG) ──▶ LLM       │
│       │          (shared, centralized)    (Llama     │
│       │                                    Stack)    │
│       ▼                                              │
│  Conversation History                                │
└─────────────────────────────────────────────────────┘
         ▲
         │ /v1/chat API
         │
┌────────┴────────────────────────────────────────────┐
│              Client-Side (not in this project)       │
│                                                      │
│  Goose Agent ──▶ Local MCP: oc/kubectl              │
│                  (user's own RBAC & kubeconfig)      │
└─────────────────────────────────────────────────────┘
```

**Why Hybrid?**
- Knowledge (RAG, docs) is **shared** — same for everyone, run once server-side
- Cluster operations (oc/kubectl) are **user-scoped** — must run with the user's RBAC

## How It Maps to the Real Lightspeed Stack

| This Project | Real LCS (lightspeed-stack) | Concept |
|---|---|---|
| `src/app/main.py` | `src/app/main.py` | FastAPI app with lifespan |
| `src/models/config.py` | `src/models/config.py` | Pydantic config validation |
| `src/models/requests.py` | `src/models/requests.py` | Request body validation |
| `src/app/endpoints/chat.py` | `src/app/endpoints/responses.py` | Main chat/responses endpoint |
| `src/app/endpoints/knowledge.py` | `src/app/endpoints/rags.py` | RAG search endpoint |
| `src/services/llm_client.py` | `src/client.py` | Llama Stack client wrapper |
| `src/services/conversation.py` | `src/cache/` | Conversation history |
| `src/middleware/auth.py` | `src/authentication/` | Auth middleware |
| `config.yaml` | `lightspeed-stack.yaml` | YAML configuration |

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 2. Run the server (uses defaults, no config file needed)
cd src && uvicorn app.main:app --reload --port 8080

# 3. Open the interactive API docs
open http://localhost:8080/docs
```

## Try It Out

### Chat with the assistant
```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Why is my pod crashlooping?"}'
```

### Search the knowledge base directly
```bash
curl -X POST http://localhost:8080/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query": "memory limit OOMKilled", "top_k": 3}'
```

### Multi-turn conversation
```bash
# First message — get a conversation_id
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I check pod resource usage?"}')

echo $RESPONSE | python3 -m json.tool

# Follow-up — use the conversation_id
CONV_ID=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['conversation_id'])")

curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What if it shows OOMKilled?\", \"conversation_id\": \"$CONV_ID\"}"
```

### Health check
```bash
curl http://localhost:8080/healthz
```

## With a Config File
```bash
APP_CONFIG_PATH=config.yaml cd src && uvicorn app.main:app --reload --port 8080
```

## Key Learning Topics

### 1. FastAPI
- **Routers**: Modular endpoint groups (`src/app/endpoints/`)
- **Lifespan**: Startup/shutdown lifecycle (`src/app/main.py`)
- **Dependency Injection**: Auth via `Depends()` (`src/middleware/auth.py`)
- **Auto-docs**: Visit `/docs` for Swagger UI, `/redoc` for ReDoc

### 2. Pydantic
- **Validation**: Automatic request/config validation (`src/models/`)
- **Field constraints**: `min_length`, `ge`, `le` (`src/models/requests.py`)
- **Custom validators**: `@field_validator`, `@model_validator` (`src/models/config.py`)
- **Serialization**: Auto JSON serialization of response models

### 3. Llama Stack
- **Client**: HTTP client for Llama Stack inference (`src/services/llm_client.py`)
- **Message format**: system/user/assistant message roles
- **RAG integration**: Context injection into prompts

### 4. Lightspeed Stack Architecture
- **Option C Hybrid**: Shared knowledge server-side, user ops client-side
- **RAG pipeline**: Retrieve → Augment → Generate (`src/app/endpoints/chat.py`)
- **Multi-turn**: Conversation history management (`src/services/conversation.py`)

## Project Structure

```
openshift-chat-assistant/
├── config.yaml                    # App configuration (like lightspeed-stack.yaml)
├── pyproject.toml                 # Dependencies and project metadata
├── README.md
└── src/
    ├── app/
    │   ├── main.py                # FastAPI app entry point
    │   └── endpoints/
    │       ├── health.py          # Health check (liveness/readiness probes)
    │       ├── chat.py            # Main chat endpoint (like /v1/responses)
    │       └── knowledge.py       # Knowledge base search (RAG retrieval)
    ├── models/
    │   ├── config.py              # Pydantic configuration models
    │   ├── requests.py            # API request models
    │   └── responses.py           # API response models
    ├── services/
    │   ├── llm_client.py          # Llama Stack client wrapper
    │   ├── knowledge.py           # Knowledge base / RAG service
    │   └── conversation.py        # Conversation history store
    └── middleware/
        └── auth.py                # Authentication middleware
```
