"""Start the REST API server."""

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime

from api import router
from config import settings
from services.container import logger

app = FastAPI(title="LangChain Agent RAG API", version="1.0.0", docs_url="/docs")
app.include_router(router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"error": "internal_error", "timestamp": datetime.utcnow().isoformat()})


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level=settings.log_level.lower())
