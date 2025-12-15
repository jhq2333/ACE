"""Agentic Context Engineering (ACE) reproduction framework."""

from .playbook import Bullet, Playbook
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .roles import (
    Generator,
    ReplayGenerator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    SimpleEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

# Import optional feature detection
from .features import has_opik, has_litellm

# Import observability components if available
if has_opik():
    try:
        from .observability import OpikIntegration
        OBSERVABILITY_AVAILABLE = True
    except ImportError:
        OpikIntegration = None
        OBSERVABILITY_AVAILABLE = False
else:
    OpikIntegration = None
    OBSERVABILITY_AVAILABLE = False

# Import production LLM clients if available
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient
        LITELLM_AVAILABLE = True
    except ImportError:
        LiteLLMClient = None
        LITELLM_AVAILABLE = False
else:
    LiteLLMClient = None
    LITELLM_AVAILABLE = False

__all__ = [
    "Bullet",
    "Playbook",
    "DeltaOperation",
    "DeltaBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    "Generator",
    "ReplayGenerator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    "OpikIntegration",
    "LITELLM_AVAILABLE",
    "OBSERVABILITY_AVAILABLE",
]
