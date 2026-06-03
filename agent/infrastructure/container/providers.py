"""Factory functions for creating provider instances.

Pure factories — no side effects, no global state mutation.
Call these from the container wiring layer.
"""

from config import settings


def _create_llm_providers():
    """Create and return the list of configured LLM providers."""
    from infrastructure.llm import (
        GoogleProvider,
        OpenAIProvider,
        OpenRouterProvider,
    )

    return [
        p for p in [
            GoogleProvider.from_settings(),
            OpenRouterProvider.from_settings(),
            OpenAIProvider.from_settings(),
        ] if p is not None
    ]


def _create_alert_providers():
    """Create and return the configured alert service."""
    from infrastructure.alerts import (
        DiscordAlertProvider,
        MultiAlertProvider,
        SlackAlertProvider,
    )

    providers = [
        p for p in [
            DiscordAlertProvider.from_settings(),
            SlackAlertProvider.from_settings(),
        ] if p is not None
    ]

    if len(providers) > 1:
        return MultiAlertProvider(providers)
    if len(providers) == 1:
        return providers[0]
    return DiscordAlertProvider(
        webhook_url=None,
        rate_limit_per_minute=settings.alert_rate_limit_per_minute,
    )


def _create_agent(llm_provider, decision_tracker, vector_store, embeddings):
    """Create the agent instance based on settings."""
    from domain.agents import RAGChainAgent
    from domain.core.chain import RAGChain
    from domain.retrieval.retriever import Retriever
    from infrastructure.agent import ToolCallingAgent

    retriever = Retriever(vector_store=vector_store, embeddings=embeddings)

    if settings.use_tool_agent:
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
        ), search_artifact_store

    chain = RAGChain(retriever=retriever, llm=llm_provider, decision_tracker=decision_tracker)
    return RAGChainAgent(chain=chain), None
