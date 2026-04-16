"""Knowledge base service — simplified RAG (Retrieval-Augmented Generation).

LEARNING NOTES:
- Real RAG systems use vector embeddings + similarity search (FAISS, Pinecone).
- Our simplified version uses keyword matching to teach the concept.
- In the Lightspeed Stack architecture (Option C - Hybrid), the knowledge
  base runs SERVER-SIDE in LCS because it's shared data — same docs for
  every developer on the cluster. No need to replicate it per workstation.
- Compare with LCS: the file_search tool and RAG vector stores.

HOW RAG WORKS (conceptually):
1. Documents are split into chunks
2. Each chunk is converted to a vector embedding (a list of numbers)
3. When a user asks a question, the question is also embedded
4. We find the chunks whose vectors are most similar (cosine similarity)
5. Those chunks are injected into the LLM prompt as context
6. The LLM generates an answer grounded in the retrieved knowledge
"""

import logging
from difflib import SequenceMatcher

from models.responses import KnowledgeChunk

logger = logging.getLogger(__name__)

# Pre-loaded knowledge about OpenShift — in a real system, these would be
# stored in a vector database and retrieved via embedding similarity search.
OPENSHIFT_KNOWLEDGE: list[dict[str, str]] = [
    {
        "source": "KB-001: Pod Lifecycle",
        "content": (
            "Pods in OpenShift go through phases: Pending, Running, Succeeded, Failed, "
            "and Unknown. A pod in CrashLoopBackOff means the container keeps crashing "
            "and Kubernetes is backing off before restarting it. Common causes: OOMKilled "
            "(out of memory), missing config/secrets, failing health probes, or application "
            "errors. Use 'oc logs <pod>' and 'oc describe pod <pod>' to diagnose."
        ),
    },
    {
        "source": "KB-002: Resource Limits",
        "content": (
            "OpenShift resource limits control CPU and memory allocation per container. "
            "If a container exceeds its memory limit, it gets OOMKilled. Set requests "
            "(guaranteed minimum) and limits (maximum allowed) in the deployment spec. "
            "Example: resources.limits.memory: 512Mi, resources.requests.memory: 256Mi. "
            "Use 'oc adm top pods' to see actual resource usage."
        ),
    },
    {
        "source": "KB-003: Routes and Services",
        "content": (
            "In OpenShift, a Service provides internal load balancing across pods. "
            "A Route exposes a Service externally with a hostname. Routes support TLS "
            "termination (edge, passthrough, re-encrypt). To create a route: "
            "'oc expose service/<name>'. To check routes: 'oc get routes'. "
            "If a route returns 503, check that the backing pods are running and "
            "the service selector matches the pod labels."
        ),
    },
    {
        "source": "KB-004: Deployments and Rollouts",
        "content": (
            "OpenShift uses Deployments (or DeploymentConfigs) to manage pod replicas. "
            "A rollout creates new ReplicaSets with updated pod specs. Strategies: "
            "RollingUpdate (default, zero-downtime) and Recreate (all-at-once). "
            "Use 'oc rollout status deployment/<name>' to monitor. "
            "Use 'oc rollout undo deployment/<name>' to rollback to the previous version."
        ),
    },
    {
        "source": "KB-005: ConfigMaps and Secrets",
        "content": (
            "ConfigMaps store non-sensitive configuration (env vars, config files). "
            "Secrets store sensitive data (passwords, tokens) base64-encoded. "
            "Mount them as environment variables or volumes in pods. "
            "If a pod fails to start with 'CreateContainerConfigError', check that "
            "all referenced ConfigMaps and Secrets exist in the namespace. "
            "Use 'oc get configmaps' and 'oc get secrets' to list them."
        ),
    },
    {
        "source": "KB-006: RBAC and Permissions",
        "content": (
            "OpenShift RBAC controls who can do what in which namespace. Key resources: "
            "Roles (namespace-scoped), ClusterRoles (cluster-wide), RoleBindings, "
            "ClusterRoleBindings. Common roles: admin, edit, view. "
            "Use 'oc auth can-i <verb> <resource>' to check permissions. "
            "Use 'oc adm policy add-role-to-user edit <user> -n <namespace>' to grant access."
        ),
    },
    {
        "source": "KB-007: Persistent Storage",
        "content": (
            "PersistentVolumeClaims (PVCs) request storage from the cluster. "
            "PersistentVolumes (PVs) are the actual storage provisioned by admins or "
            "dynamically via StorageClasses. Access modes: ReadWriteOnce (RWO), "
            "ReadOnlyMany (ROX), ReadWriteMany (RWX). If a PVC stays Pending, "
            "check StorageClass availability and capacity. "
            "Use 'oc get pvc' and 'oc describe pvc <name>' to diagnose."
        ),
    },
    {
        "source": "KB-008: Networking and Network Policies",
        "content": (
            "OpenShift uses an SDN (Software Defined Network) for pod networking. "
            "By default, all pods can communicate. NetworkPolicies restrict traffic "
            "between pods based on labels, namespaces, and ports. "
            "Use 'oc get networkpolicy' to list policies. "
            "If pods can't communicate, check NetworkPolicies and Service selectors."
        ),
    },
]


def _similarity(query: str, text: str) -> float:
    """Compute a simple similarity score between query and text.

    LEARNING: Real systems use cosine similarity between embedding vectors.
    We use SequenceMatcher + keyword overlap as a teaching approximation.
    """
    query_lower = query.lower()
    text_lower = text.lower()

    # Keyword overlap score
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    if not query_words:
        return 0.0
    keyword_score = len(query_words & text_words) / len(query_words)

    # Sequence similarity
    seq_score = SequenceMatcher(None, query_lower, text_lower[:200]).ratio()

    # Combine both signals
    return 0.6 * keyword_score + 0.4 * seq_score


def search_knowledge(query: str, top_k: int = 3, threshold: float = 0.3) -> list[KnowledgeChunk]:
    """Search the knowledge base for relevant chunks.

    This is the core RAG retrieval step. In a real system:
    1. The query would be embedded using the same model that embedded the docs
    2. A vector DB would return the nearest neighbors
    3. Results above a similarity threshold would be returned

    Args:
        query: The user's question.
        top_k: Maximum number of results to return.
        threshold: Minimum similarity score to include.

    Returns:
        List of relevant KnowledgeChunks sorted by score descending.
    """
    results: list[KnowledgeChunk] = []

    for doc in OPENSHIFT_KNOWLEDGE:
        score = _similarity(query, doc["content"])
        if score >= threshold:
            results.append(
                KnowledgeChunk(
                    content=doc["content"],
                    source=doc["source"],
                    score=round(score, 3),
                )
            )

    # Sort by score descending, take top_k
    results.sort(key=lambda x: x.score, reverse=True)
    logger.info("Knowledge search for '%s': found %d results", query, len(results[:top_k]))
    return results[:top_k]
