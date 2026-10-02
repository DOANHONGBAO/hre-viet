from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from hre_translate.serving.engine import (
    ModelUnavailableError,
    ServingEngine,
    UnsupportedDirectionError,
)
from hre_translate.serving.feedback import FeedbackStore
from hre_translate.serving.schemas import (
    BatchRequest,
    BatchResponse,
    FeedbackRequest,
    FeedbackResponse,
    TranslateRequest,
    TranslateResponse,
)
from hre_translate.utils.config import load_config, project_root, resolve


def create_app(
    engine: ServingEngine | None = None,
    feedback_store: FeedbackStore | None = None,
    *,
    root: Path | None = None,
) -> FastAPI:
    root = root or project_root()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.engine = engine or ServingEngine.from_project(root)
        if feedback_store is None:
            config, _ = load_config(resolve(root, "configs/serving.yaml"))
            app.state.feedback = FeedbackStore(resolve(root, config["feedback_db"]))
        else:
            app.state.feedback = feedback_store
        yield

    app = FastAPI(title="HRE-TRANSLATE", version="0.1.0", lifespan=lifespan)
    config, _ = load_config(resolve(root, "configs/serving.yaml"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.get("frontend_origins", [])),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, object]:
        neural = app.state.engine.neural
        return {
            "status": "ok",
            "neural_available": bool(getattr(neural, "available", True)),
            "neural_loaded": bool(getattr(neural, "loaded", False)),
        }

    @app.get("/models")
    def models() -> list[dict[str, object]]:
        return app.state.engine.models()

    @app.post("/translate", response_model=TranslateResponse)
    def translate(request: TranslateRequest) -> TranslateResponse:
        try:
            return app.state.engine.translate(request)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except UnsupportedDirectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/translate/batch", response_model=BatchResponse)
    def translate_batch(request: BatchRequest) -> BatchResponse:
        started = time.perf_counter()
        try:
            results = [
                app.state.engine.translate(
                    TranslateRequest(
                        text=text,
                        source=request.source,
                        target=request.target,
                        model=request.model,
                    )
                )
                for text in request.texts
            ]
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except UnsupportedDirectionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return BatchResponse(
            translations=results,
            latency_ms_total=(time.perf_counter() - started) * 1000,
        )

    @app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
    def feedback(request: FeedbackRequest) -> FeedbackResponse:
        return app.state.feedback.save(request)

    return app


app = create_app()
