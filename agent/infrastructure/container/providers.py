"""Factory functions for creating provider instances.

Pure factories — no side effects, no global state mutation.
Call these from the container wiring layer.
"""

from typing import List, Type


def _configured(provider_classes: List[Type]) -> list:
    """Instantiate only providers that are properly configured."""
    return [cls() for cls in provider_classes if cls.is_configured()]


def _create_llm_providers():
    """Create and return the list of configured LLM providers."""
    from infrastructure.llm import (
        GoogleProvider,
        OpenAIProvider,
        OpenRouterProvider,
    )

    return _configured([GoogleProvider, OpenRouterProvider, OpenAIProvider])


def _create_alert_providers():
    """Create and return the configured alert service."""
    from infrastructure.alerts import (
        DiscordAlertProvider,
        MultiAlertProvider,
        SlackAlertProvider,
    )

    providers = _configured([
        DiscordAlertProvider,
        SlackAlertProvider,
    ])

    return MultiAlertProvider(providers)

def _create_agent(llm_provider, decision_tracker, vector_store, embeddings):
    """Create the ToolCallingAgent with search and document tools."""
    from domain.retrieval.retriever import Retriever
    from infrastructure.agent import ToolCallingAgent

    retriever = Retriever(vector_store=vector_store, embeddings=embeddings)
    search_artifact_store = []

    from infrastructure.tools import create_search_documents_tool, web_search_tool

    agent_tools = [
        create_search_documents_tool(
            retriever=retriever,
            artifact_store=search_artifact_store,
            default_latest_only=True,
        ),
        web_search_tool,
    ]

    return ToolCallingAgent(
        llm=llm_provider.chat_model,
        tools=agent_tools,
        artifact_store=search_artifact_store,
        default_top_k=5,
        decision_tracker=decision_tracker,
    )
