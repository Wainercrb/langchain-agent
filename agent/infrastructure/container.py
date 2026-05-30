"""Composition root — single place to wire all pluggable service instances.

Change the instances below to swap implementations across the whole app.

Examples:
    # LLM
    llm = GeminiProvider(model="gemini-2.5-flash", ...)
    # llm = OpenAIProvider(model="gpt-4", api_key="sk-...")

    # Vector Store
    # vector_store = PineconeVectorStore(api_key="...")
    # vector_store = QdrantVectorStore(url="...")

    # Embeddings
    # embeddings = OpenAIEmbeddingsProvider(api_key="...")
"""

from config import settings

# ── LLM ──────────────────────────────────────────────────────────────
from infrastructure.llm import OpenRouterProvider

# llm = GoogleProvider(
#     model=settings.gemini_model,
#     temperature=settings.gemini_temperature,
#     api_key=settings.google_api_key,
# )

# llm = OpenAIProvider(
#     model=settings.openai_model,
#     temperature=settings.openai_temperature,
#     api_key=settings.openai_api_key or None,
# )

llm = OpenRouterProvider(
    model=settings.openrouter_model,
    temperature=settings.openrouter_temperature,
    max_tokens=settings.openrouter_max_tokens,
    api_key=settings.openrouter_api_key,
    timeout=settings.llm_timeout_seconds,
)

# ── Embeddings ───────────────────────────────────────────────────────
from infrastructure.embeddings import GoogleEmbeddingsWrapper

embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)

# ── Vector Store ─────────────────────────────────────────────────────
from supabase import create_client
from infrastructure.vector_store import VectorStore

db_direct_url = settings.supabase_direct_url
db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── Feedback ─────────────────────────────────────────────────────────
from infrastructure.feedback import LangSmithFeedbackProvider

feedback_service = LangSmithFeedbackProvider()

# ── Alerts ────────────────────────────────────────────────────────────
from infrastructure.alerts import DiscordAlertProvider

# Future: swap DiscordAlertProvider for SlackAlertProvider or TeamsAlertProvider
alert_service = DiscordAlertProvider(
    webhook_url=settings.discord_webhook_url,
    rate_limit_per_minute=settings.alert_rate_limit_per_minute,
)

# ── Agent ─────────────────────────────────────────────────────────────
from domain.retrieval.retriever import Retriever
from domain.core.chain import RAGChain
from infrastructure.agent import ToolCallingAgent, RAGChainAgent
from infrastructure.tools import create_search_documents_tool, web_search_tool

_retriever = Retriever(vector_store=vector_store, embeddings=embeddings)

if settings.use_tool_agent:
    _search_artifact_store: list = []
    _tools = [
        create_search_documents_tool(
            retriever=_retriever,
            artifact_store=_search_artifact_store,
            default_latest_only=True,
        ),
        web_search_tool,
    ]
    agent = ToolCallingAgent(
        llm=llm.chat_model,
        tools=_tools,
        artifact_store=_search_artifact_store,
        default_top_k=5,
    )
else:
    _chain = RAGChain(retriever=_retriever, llm=llm)
    agent = RAGChainAgent(chain=_chain)
