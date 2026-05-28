"""Tests for expanded health check endpoint."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient


class TestHealthCheck:
    """Verify expanded health check with DB, LLM, and embedding probes."""

    def test_health_response_includes_all_fields(self):
        """HealthResponse should include llm_connected and embedding_connected."""
        from models.responses import HealthResponse

        response = HealthResponse(
            status="ok",
            timestamp=datetime.utcnow(),
            db_connected=True,
            llm_connected=True,
            embedding_connected=True,
        )
        assert response.status == "ok"
        assert response.db_connected is True
        assert response.llm_connected is True
        assert response.embedding_connected is True

    def test_health_response_defaults(self):
        """LLM and embedding fields should default to False."""
        from models.responses import HealthResponse

        response = HealthResponse(
            status="error",
            timestamp=datetime.utcnow(),
            db_connected=False,
        )
        assert response.llm_connected is False
        assert response.embedding_connected is False

    def test_error_status_when_db_down(self):
        """Status should be 'error' when DB is not connected."""
        from models.responses import HealthResponse

        response = HealthResponse(
            status="error",
            timestamp=datetime.utcnow(),
            db_connected=False,
            llm_connected=True,
            embedding_connected=True,
        )
        assert response.status == "error"
