import logging
from typing import Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration template for an LLM provider.
    
    Represents provider-specific settings mappings and defaults.
    
    Attributes:
        name: Provider identifier (e.g., "gemini", "openai")
        model_key: Settings attribute for model name
        temperature_key: Settings attribute for temperature
        api_key_key: Settings attribute for API key
        default_model: Fallback model if not in settings
        default_temperature: Fallback temperature if not in settings
    """
    name: str
    model_key: str
    temperature_key: str
    api_key_key: str
    default_model: str
    default_temperature: float = 0.7


class ProviderRegistry:
    """Registry for LLM provider configurations using Composite Pattern.
    
    Manages provider configurations and enables:
    - Dynamic provider registration (open for extension)
    - Type-safe configuration lookup
    - Easy provider addition without modifying get_llm()
    
    Pattern: Registry Pattern + Composite Pattern
    - Registry: centralized provider configuration management
    - Composite: collection of provider configs with uniform interface
    """
    
    def __init__(self):
        """Initialize registry with built-in providers."""
        self._providers: Dict[str, ProviderConfig] = {}
        self._register_default_providers()
    
    def _register_default_providers(self) -> None:
        """Register built-in LLM providers."""
        self.register(ProviderConfig(
            name="gemini",
            model_key="gemini_model",
            temperature_key="gemini_temperature",
            api_key_key="google_api_key",
            default_model="gemini-2.5-flash",
            default_temperature=0.7,
        ))
        
        self.register(ProviderConfig(
            name="openai",
            model_key="openai_model",
            temperature_key="openai_temperature",
            api_key_key="openai_api_key",
            default_model="gpt-4",
            default_temperature=0.7,
        ))
        
        self.register(ProviderConfig(
            name="anthropic",
            model_key="anthropic_model",
            temperature_key="anthropic_temperature",
            api_key_key="anthropic_api_key",
            default_model="claude-3-opus-20240229",
            default_temperature=0.7,
        ))
    
    def register(self, config: ProviderConfig) -> None:
        """Register a new provider configuration.
        
        Args:
            config: ProviderConfig instance to register
            
        Raises:
            ValueError: If provider with same name already registered
        """
        if config.name in self._providers:
            raise ValueError(f"Provider '{config.name}' already registered")
        self._providers[config.name] = config
        logger.debug(f"Registered provider: {config.name}")
    
    def get(self, provider_name: str) -> ProviderConfig:
        """Get provider configuration by name.
        
        Args:
            provider_name: Provider identifier
            
        Returns:
            ProviderConfig for the provider
            
        Raises:
            ValueError: If provider not found in registry
        """
        if provider_name not in self._providers:
            available = ", ".join(self._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: '{provider_name}'. "
                f"Available: {available}. "
                f"To add a new provider: registry.register(ProviderConfig(...))"
            )
        return self._providers[provider_name]
    
    def list_providers(self) -> list:
        """Get list of all registered provider names.
        
        Returns:
            List of provider identifiers
        """
        return list(self._providers.keys())
    
    def has_provider(self, provider_name: str) -> bool:
        """Check if provider is registered.
        
        Args:
            provider_name: Provider identifier
            
        Returns:
            True if provider registered, False otherwise
        """
        return provider_name in self._providers


# Global registry singleton
_provider_registry = ProviderRegistry()


class Services:
    """
    Singleton service registry for shared RAG dependencies.

    Manages lazy initialization and caching of expensive resources:
    - VectorStore: Supabase connection with pgvector embeddings
    - GoogleEmbeddingsWrapper: Embedding model for vector generation
    - ChatGoogleGenerativeAI: LLM for response generation

    Pattern: Services are initialized on first request, then reused across
    all subsequent requests to minimize connection overhead and latency.
    """

    vector_store: Optional[object] = None
    embeddings: Optional[object] = None
    llm: Optional[object] = None


def get_vector_store():
    """
    Get or initialize VectorStore singleton.

    Lazily initializes connection to Supabase with pgvector support.
    On first call, creates connection using credentials from config.Settings.

    Returns:
        VectorStore: Connected vector store instance for document retrieval

    Raises:
        Exception: If Supabase connection fails or credentials missing

    Note:
        Connection is cached. Subsequent calls return the same instance.
    """
    if Services.vector_store is None:
        try:
            from supabase import create_client
            from config import Settings
            from rag.retrieval.vector_store import VectorStore

            settings = Settings()
            supabase_client = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
            Services.vector_store = VectorStore(supabase_client)
            logger.info("VectorStore singleton initialized")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}", exc_info=True)
            raise

    return Services.vector_store


def get_embeddings():
    """
    Get or initialize GoogleEmbeddingsWrapper singleton.

    Lazily initializes Google Generative AI embeddings model.
    On first call, creates wrapper using API key from config.Settings.

    Returns:
        GoogleEmbeddingsWrapper: Embedding model instance for document vectorization

    Raises:
        Exception: If Google API key missing or initialization fails

    Note:
        Wrapper is cached. Subsequent calls return the same instance.
    """
    if Services.embeddings is None:
        try:
            from config import Settings
            from rag.embeddings.wrapper import GoogleEmbeddingsWrapper

            settings = Settings()
            Services.embeddings = GoogleEmbeddingsWrapper(api_key=settings.google_api_key)
            logger.info("GoogleEmbeddingsWrapper singleton initialized")
        except Exception as e:
            logger.error(
                f"Failed to initialize GoogleEmbeddingsWrapper: {str(e)}",
                exc_info=True,
            )
            raise

    return Services.embeddings


def get_llm():
    """
    Get or initialize LLM provider singleton using OOP Provider Registry Pattern.

    Lazily initializes the LLM provider (Gemini, OpenAI, Anthropic) based on config.
    Uses ProviderRegistry (type-safe OOP) to dynamically fetch configuration.
    Adding new providers requires only calling registry.register(ProviderConfig(...)).
    
    Design Patterns:
    - Registry Pattern: centralized provider config management
    - Composite Pattern: uniform interface for all providers
    - Strategy Pattern: all providers implement LLMProvider interface
    - Lazy Initialization: provider created on first use, cached

    Returns:
        LLMProvider: Initialized LLM provider instance

    Raises:
        ValueError: If provider not in registry
        LLMProviderError: If provider initialization fails or credentials missing

    Note:
        Provider is cached. Subsequent calls return the same instance.
        
    Example - Add new provider at runtime:
        from api.dependencies import _provider_registry, ProviderConfig
        
        # Add Cohere provider
        _provider_registry.register(ProviderConfig(
            name="cohere",
            model_key="cohere_model",
            temperature_key="cohere_temperature",
            api_key_key="cohere_api_key",
            default_model="command-r-plus",
            default_temperature=0.7,
        ))
        
        # Now you can use it
        export LLM_PROVIDER=cohere
        python main.py
    """
    if Services.llm is None:
        try:
            from config import settings
            from services import create_llm_provider

            # Get provider name from configuration
            provider_name = settings.llm_provider
            
            # Get provider config from registry (OOP approach)
            provider_config = _provider_registry.get(provider_name)
            
            # Dynamically fetch provider-specific settings using config object
            model = (
                getattr(settings, provider_config.model_key, None) 
                or provider_config.default_model
            )
            temperature = (
                getattr(settings, provider_config.temperature_key, None) 
                or provider_config.default_temperature
            )
            api_key = getattr(settings, provider_config.api_key_key)

            # Create provider using factory (Strategy pattern)
            Services.llm = create_llm_provider(
                provider_name=provider_name,
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
            logger.info(
                f"LLM provider singleton initialized: "
                f"{Services.llm.get_provider_name()} (model={model})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {str(e)}", exc_info=True)
            raise

    return Services.llm


def get_retriever():
    """
    Create a new Retriever instance for document retrieval.

    Composes VectorStore and GoogleEmbeddingsWrapper for similarity-based retrieval.
    A new instance is created per request to ensure stateless operation, though
    underlying services (VectorStore, embeddings) are cached singletons.

    Returns:
        Retriever: Document retrieval pipeline instance

    Raises:
        Exception: If VectorStore or embeddings initialization fails

    Note:
        New instance per request ensures thread-safety and request isolation.
    """
    try:
        from rag.retrieval.retriever import Retriever

        retriever = Retriever(
            vector_store=get_vector_store(),
            embeddings=get_embeddings(),
        )
        logger.debug("Retriever instance created for request")
        return retriever
    except Exception as e:
        logger.error(f"Failed to create Retriever: {str(e)}", exc_info=True)
        raise


def get_rag_chain():
    """
    Create a new RAGChain instance for query processing.

    Composes Retriever and LLM for full RAG pipeline:
    1. Retrieve relevant documents via similarity search
    2. Construct context-aware prompt
    3. Generate response via LLM

    Returns:
        RAGChain: Complete RAG pipeline instance

    Raises:
        Exception: If Retriever or LLM initialization fails

    Note:
        New instance per request ensures request isolation.
        Underlying singletons (VectorStore, Embeddings, LLM) are reused.
    """
    try:
        from rag.core.chain import RAGChain

        chain = RAGChain(
            retriever=get_retriever(),
            llm=get_llm(),
        )
        logger.debug("RAGChain instance created for request")
        return chain
    except Exception as e:
        logger.error(f"Failed to create RAGChain: {str(e)}", exc_info=True)
        raise


async def check_health() -> dict:
    """
    Perform system health check including database connectivity.

    Validates core service availability:
    - VectorStore connection to Supabase
    - Database responsiveness
    - Overall system readiness

    Returns:
        dict containing:
            - status: "ok" if healthy, "error" if degraded
            - db_connected: Boolean database connectivity status
            - error: (optional) Error message if status is "error"

    Note:
        This function gracefully handles errors, returning degraded status
        rather than raising exceptions. Clients can assess partial system health.
    """
    try:
        logger.debug("Performing health check")
        vector_store = get_vector_store()

        result = await vector_store.health_check()
        db_status = "ok" if result else "error"

        logger.info(f"Health check complete: status={db_status}")
        return {
            "status": db_status,
            "db_connected": bool(result),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "db_connected": False,
            "error": str(e),
        }
