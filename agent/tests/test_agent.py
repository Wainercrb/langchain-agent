"""Unit tests for the Agent strategy implementations.

Tests both ToolCallingAgent and RAGChainAgent with mocked components.
Verifies the Agent ABC contract and strategy swapability.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from models import ChatResponse, RetrievedDocument, SourceDocument
from services.agent.base import Agent


# ═══════════════════════════════════════════════════════════════════════
#  Fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_retriever():
    """Mock Retriever with sample documents."""
    retriever = Mock()
    retriever.retrieve.return_value = [
        RetrievedDocument(
            document_id="doc1",
            chunk_id="chunk1",
            text="How to enroll: step 1 is to complete the online form.",
            similarity_score=0.95,
            filename="enrollment_guide.pdf",
            version_date=datetime(2025, 1, 1),
        ),
        RetrievedDocument(
            document_id="doc2",
            chunk_id="chunk2",
            text="Enrollment requirements: valid government ID.",
            similarity_score=0.88,
            filename="requirements.txt",
            version_date=datetime(2025, 1, 2),
        ),
    ]
    return retriever


@pytest.fixture
def mock_chat_model():
    """Mock LangChain chat model that simulates tool calling."""
    model = MagicMock()

    def bind_tools_side_effect(tools, **kwargs):
        return model

    model.bind_tools = MagicMock(side_effect=bind_tools_side_effect)
    model.model = "mock-model"
    return model


@pytest.fixture
def sample_tools(mock_retriever):
    """Build a minimal tool list for testing."""
    from services.tools import create_search_documents_tool

    artifact_store = []
    search_tool = create_search_documents_tool(
        retriever=mock_retriever,
        artifact_store=artifact_store,
    )
    return [search_tool]


# ═══════════════════════════════════════════════════════════════════════
#  ABC Contract Tests
# ═══════════════════════════════════════════════════════════════════════


class TestAgentABC:
    """Verify the Agent abstract base class contract."""

    def test_agent_is_abstract(self):
        """Direct instantiation of Agent should fail."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Agent()

    def test_tool_calling_agent_is_agent(self, mock_chat_model, sample_tools):
        """ToolCallingAgent satisfies the Agent ABC."""
        from services.agent.tool_calling import ToolCallingAgent

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)
        assert isinstance(agent, Agent)

    def test_rag_chain_agent_is_agent(self):
        """RAGChainAgent satisfies the Agent ABC."""
        from services.agent.rag_chain import RAGChainAgent
        from rag.core.chain import RAGChain

        mock_chain = Mock(spec=RAGChain)
        agent = RAGChainAgent(chain=mock_chain)
        assert isinstance(agent, Agent)


# ═══════════════════════════════════════════════════════════════════════
#  ToolCallingAgent Tests
# ═══════════════════════════════════════════════════════════════════════


class TestToolCallingAgentInvoke:
    """Tests for ToolCallingAgent.invoke with mocked executor."""

    @patch("services.agent.tool_calling.create_tool_calling_agent")
    @patch("services.agent.tool_calling.AgentExecutor")
    def test_invoke_returns_chat_response(
        self,
        mock_executor_class,
        mock_create_agent,
        mock_chat_model,
        sample_tools,
    ):
        """invoke() returns a ChatResponse with expected fields."""
        from services.agent.tool_calling import ToolCallingAgent

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "This is a test response."}
        mock_executor_class.return_value = mock_executor
        mock_create_agent.return_value = MagicMock()

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)
        response = agent.invoke(query="test query", top_k=5)

        assert isinstance(response, ChatResponse)
        assert response.response == "This is a test response."
        assert response.query == "test query"
        assert response.execution_time_ms > 0
        assert response.model == "mock-model"
        assert response.run_id is not None

    @patch("services.agent.tool_calling.create_tool_calling_agent")
    @patch("services.agent.tool_calling.AgentExecutor")
    def test_invoke_without_sources(
        self,
        mock_executor_class,
        mock_create_agent,
        mock_chat_model,
        sample_tools,
    ):
        """Sources are None when include_sources=False."""
        from services.agent.tool_calling import ToolCallingAgent

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Direct answer."}
        mock_executor_class.return_value = mock_executor
        mock_create_agent.return_value = MagicMock()

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)
        response = agent.invoke(query="What is 2+2?", include_sources=False)

        assert response.sources is None

    @patch("services.agent.tool_calling.create_tool_calling_agent")
    @patch("services.agent.tool_calling.AgentExecutor")
    def test_invoke_error_handling(
        self,
        mock_executor_class,
        mock_create_agent,
        mock_chat_model,
        sample_tools,
    ):
        """Agent errors propagate correctly."""
        from services.agent.tool_calling import ToolCallingAgent

        mock_executor = MagicMock()
        mock_executor.invoke.side_effect = Exception("Agent execution failed")
        mock_executor_class.return_value = mock_executor
        mock_create_agent.return_value = MagicMock()

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)

        with pytest.raises(Exception, match="Agent execution failed"):
            agent.invoke(query="test")

    @patch("services.agent.tool_calling.create_tool_calling_agent")
    @patch("services.agent.tool_calling.AgentExecutor")
    def test_invoke_query_echo(
        self,
        mock_executor_class,
        mock_create_agent,
        mock_chat_model,
        sample_tools,
    ):
        """Query is echoed back in the response."""
        from services.agent.tool_calling import ToolCallingAgent

        mock_executor = MagicMock()
        mock_executor.invoke.return_value = {"output": "Answer"}
        mock_executor_class.return_value = mock_executor
        mock_create_agent.return_value = MagicMock()

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)
        response = agent.invoke(query="What are the requirements?")

        assert response.query == "What are the requirements?"


class TestToolCallingAgentSources:
    """Tests for source extraction from tool artifact stores."""

    def test_extracts_sources_from_search_tool(self, mock_retriever):
        """Agent extracts sources when search_documents tool has artifacts."""
        from services.agent.tool_calling import ToolCallingAgent
        from services.tools import create_search_documents_tool

        artifact_store = []
        search_tool = create_search_documents_tool(
            retriever=mock_retriever, artifact_store=artifact_store
        )

        agent = ToolCallingAgent(
            llm=MagicMock(),
            tools=[search_tool],
        )

        # Simulate tool invocation populating the store
        search_tool.invoke({"query": "enrollment", "top_k": 5})
        assert len(artifact_store) == 2

        sources = agent._extract_sources(include_sources=True)
        assert sources is not None
        assert len(sources) == 2
        assert sources[0].document_id == "doc1"
        assert sources[0].filename == "enrollment_guide.pdf"

        # Store should be cleared after extraction
        assert len(getattr(search_tool, "artifact_store", [])) == 0

    def test_no_sources_when_include_sources_false(self, mock_retriever):
        """Agent skips extraction when include_sources=False."""
        from services.agent.tool_calling import ToolCallingAgent
        from services.tools import create_search_documents_tool

        artifact_store = []
        search_tool = create_search_documents_tool(
            retriever=mock_retriever, artifact_store=artifact_store
        )
        search_tool.invoke({"query": "test", "top_k": 5})

        agent = ToolCallingAgent(
            llm=MagicMock(),
            tools=[search_tool],
        )

        sources = agent._extract_sources(include_sources=False)
        assert sources is None


class TestToolCallingAgentConfiguration:
    """Tests for agent configuration and tool injection."""

    def test_prompt_includes_tool_descriptions(self, mock_chat_model, sample_tools):
        """System prompt dynamically lists injected tool names."""
        from services.agent.tool_calling import ToolCallingAgent

        agent = ToolCallingAgent(llm=mock_chat_model, tools=sample_tools)
        prompt = agent._build_prompt()

        # The prompt should mention the tool name
        assert "search_documents" in prompt.messages[0].prompt.template

    def test_default_top_k(self, mock_chat_model, sample_tools):
        """default_top_k is configurable."""
        from services.agent.tool_calling import ToolCallingAgent

        agent = ToolCallingAgent(
            llm=mock_chat_model, tools=sample_tools, default_top_k=10
        )
        assert agent._default_top_k == 10


# ═══════════════════════════════════════════════════════════════════════
#  RAGChainAgent Tests
# ═══════════════════════════════════════════════════════════════════════


class TestRAGChainAgent:
    """Tests for the legacy RAGChainAgent adapter."""

    def test_delegates_to_rag_chain(self):
        """RAGChainAgent delegates invoke() to the wrapped chain."""
        from services.agent.rag_chain import RAGChainAgent

        mock_chain = Mock()
        mock_chain.invoke.return_value = ChatResponse(
            response="Legacy answer",
            query="test",
            execution_time_ms=50.0,
            model="legacy-model",
        )

        agent = RAGChainAgent(chain=mock_chain)
        response = agent.invoke(query="test", top_k=3, temperature=0.5)

        mock_chain.invoke.assert_called_once_with(
            query="test", top_k=3, temperature=0.5, include_sources=True, latest_only=True
        )
        assert response.response == "Legacy answer"

    def test_satisfies_agent_abc(self):
        """RAGChainAgent is an Agent."""
        from services.agent.rag_chain import RAGChainAgent

        mock_chain = Mock()
        agent = RAGChainAgent(chain=mock_chain)
        assert isinstance(agent, Agent)


# ═══════════════════════════════════════════════════════════════════════
#  Response Schema Tests
# ═══════════════════════════════════════════════════════════════════════


class TestChatResponseSchema:
    """Ensure ChatResponse schema matches API contract."""

    def test_serialization(self):
        """ChatResponse serializable to dict with all required fields."""
        response = ChatResponse(
            response="Test answer",
            query="Test query",
            sources=None,
            execution_time_ms=100.0,
            model="test-model",
            run_id="test-run-id",
        )

        data = response.model_dump()
        assert data["response"] == "Test answer"
        assert data["query"] == "Test query"
        assert data["sources"] is None
        assert data["execution_time_ms"] == 100.0
        assert data["model"] == "test-model"
        assert data["run_id"] == "test-run-id"

    def test_with_sources_serialization(self):
        """ChatResponse with sources serializes correctly."""
        sources = [
            SourceDocument(
                document_id="doc1",
                filename="test.pdf",
                similarity_score=0.95,
                version_date=datetime(2025, 1, 1),
                content_preview="Test content preview...",
                chunk_id="chunk1",
            )
        ]

        response = ChatResponse(
            response="Test answer",
            query="Test query",
            sources=sources,
            execution_time_ms=100.0,
            model="test-model",
        )

        data = response.model_dump()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_id"] == "doc1"
        assert data["sources"][0]["filename"] == "test.pdf"
        assert data["sources"][0]["similarity_score"] == 0.95

    def test_execution_time_non_negative(self):
        """execution_time_ms must be >= 0."""
        response = ChatResponse(
            response="Answer", query="Query", execution_time_ms=0.0, model="test"
        )
        assert response.execution_time_ms >= 0
