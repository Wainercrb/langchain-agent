"""Thin router — imports all handler routers and includes them."""

from fastapi import APIRouter

from api.handlers import (
    chat_router,
    circuits_router,
    decisions_router,
    feedback_router,
    health_router,
    metrics_router,
    monitoring_router,
)

router = APIRouter()
router.include_router(chat_router)
router.include_router(circuits_router)
router.include_router(decisions_router)
router.include_router(feedback_router)
router.include_router(health_router)
router.include_router(metrics_router)
router.include_router(monitoring_router)
