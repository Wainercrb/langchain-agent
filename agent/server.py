"""Start the REST API server."""

import uvicorn
from fastapi import FastAPI

from api import router
from api.exception_handlers import global_exception_handler
from api.lifespan import lifespan
from api.middleware_config import configure_correlation_middleware, configure_middleware
from config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="LangChain Agent RAGAPI", version="1.0.0", docs_url="/docs", lifespan=lifespan)

    configure_middleware(app)
    configure_correlation_middleware(app)
    app.include_router(router)
    app.exception_handler(Exception)(global_exception_handler)

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
