"""Errores de dominio y manejadores que devuelven respuestas JSON uniformes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("ctr.api")


class AppError(Exception):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error = "internal_error"

    def __init__(self, detail: Any = None) -> None:
        super().__init__(detail)
        self.detail = detail


class ModelNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error = "model_not_found"


class NoModelsAvailableError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error = "no_models_available"


class BatchTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error = "batch_too_large"


class PredictionError(AppError):
    error = "prediction_failed"


class ResourceNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error = "resource_not_found"


def _body(request: Request, error: str, detail: Any) -> dict[str, Any]:
    return {"error": error, "detail": detail, "request_id": getattr(request.state, "request_id", None)}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(_body(request, exc.error, exc.detail), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [
            {"field": ".".join(str(p) for p in err["loc"] if p != "body") or "body", "message": err["msg"],
             "type": err["type"]}
            for err in exc.errors()
        ]
        return JSONResponse(_body(request, "validation_error", detail),
                            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        error = "not_found" if exc.status_code == 404 else "http_error"
        return JSONResponse(_body(request, error, exc.detail), status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        log.exception("Error no controlado en %s %s", request.method, request.url.path)
        return JSONResponse(_body(request, "internal_error", "Error interno del servidor"),
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
