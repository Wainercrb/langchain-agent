"""Pydantic models for circuit breaker status responses."""

from typing import Dict

from pydantic import BaseModel


class CircuitStatusResponse(BaseModel):
    """Response model for GET /v1/llm/circuits."""

    circuits: Dict[str, str]  # provider_name -> circuit state
