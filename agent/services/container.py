"""Composition root — single place to wire all pluggable service instances.

Cambiá las instancias de abajo para cambiar implementaciones en toda la app.

Ejemplos:
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

# ── Logger (creado primero, antes de imports de servicios) ────────────
from services.logging import Console
logger = Console()

# ── LLM ──────────────────────────────────────────────────────────────
from services.llm import GoogleProvider, OpenAIProvider, OpenRouterProvider

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
from services.embeddings import GoogleEmbeddingsWrapper

embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)

# ── Vector Store ─────────────────────────────────────────────────────
from supabase import create_client
from services.vector_store import VectorStore

db_direct_url = settings.supabase_direct_url
db_url = settings.supabase_url
db_key = settings.supabase_key
supabase_client = create_client(db_url, db_key)
vector_store = VectorStore(supabase_client)

# ── LangSmith Tracing ──────────────────────────────────────────────
tracing_callback = None
if settings.langchain_tracing_v2 and settings.langsmith_api_key:
    try:
        from langchain.callbacks.tracers.langchain import LangChainTracer
        tracing_callback = LangChainTracer(project=settings.langchain_project)
        logger.info(f"LangSmith tracing enabled: project={settings.langchain_project}")
    except ImportError:
        logger.warning("langsmith package not installed — tracing disabled")
    except Exception as e:
        logger.warning(f"LangSmith tracer init failed: {str(e)}")
