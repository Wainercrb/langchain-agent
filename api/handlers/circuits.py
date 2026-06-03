"""Circuit breaker status endpoint — LLM provider circuit states."""

from fastapi import APIRouter

from infrastructure.container import llm
from models import CircuitStatusResponse

router = APIRouter(prefix="/v1", tags=["llm"])


@router.get(
    "/llm/circuits",
    response_model=CircuitStatusResponse,
    status_code=200,
)
async def circuit_status() -> CircuitStatusResponse:
    """Return current circuit breaker status for all LLM providers.

    Returns:
        CircuitStatusResponse with provider name -> circuit state mapping.
    """
    return CircuitStatusResponse(
        circuits=llm.circuit_status() if hasattr(llm, "circuit_status") else {},
    )
