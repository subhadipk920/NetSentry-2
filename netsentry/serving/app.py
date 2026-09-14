"""
NetSentry FastAPI Serving Application (`netsentry.serving.app`).
----------------------------------------------------------------
Initializes model loader lifecycle, registers prediction and health routes,
and exposes the hot-reload endpoint.
"""

from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
load_dotenv()

from netsentry.serving.config import ServingConfig, load_serving_config
from netsentry.serving.dependencies import get_predictor
from netsentry.serving.health import router as health_router
from netsentry.serving.model_loader import ModelLoader
from netsentry.serving.predictor import Predictor
from netsentry.serving.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger("NetSentryServing")


def create_app(config: Optional[ServingConfig] = None, model_loader: Optional[ModelLoader] = None) -> FastAPI:
    """Factory creating and configuring the NetSentry serving application."""
    srv_cfg = config or load_serving_config()
    loader = model_loader or ModelLoader(config=srv_cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: Load champion model and threshold from MLflow Model Registry
        try:
            loader.load()
            app.state.model_loader = loader
            logger.info(
                f"NetSentry Champion loaded: version={loader.champion_model.version}, "
                f"threshold={loader.champion_model.threshold}"
            )
        except Exception as e:
            logger.warning(f"Could not load champion model on startup: {e}")
            app.state.model_loader = loader

        yield

        # Shutdown: Clear app state
        app.state.model_loader = None

    app = FastAPI(
        title=srv_cfg.api_title,
        version=srv_cfg.api_version,
        lifespan=lifespan,
    )

    # Health & readiness routes
    app.include_router(health_router)

    # Security Gateway Reverse Proxy
    from netsentry.serving.routes.gateway import router as gateway_router
    app.include_router(gateway_router)

    # Prediction router
    @app.get("/")
    async def read_root():
        return {"message": "Welcome to the NetSentry API!"}

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
    @app.post("/v2/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(
        request: PredictionRequest,
        predictor: Predictor = Depends(get_predictor),
    ) -> PredictionResponse:
        try:
            pred, prob = predictor.predict(request.features)
            return PredictionResponse(
                prediction=pred,
                probability=prob,
                model_version=predictor.champion.version,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Inference error: {str(e)}",
            )

    @app.post("/v1/model/reload", tags=["Lifecycle"])
    async def reload_model(request: Request):
        ldr: ModelLoader = request.app.state.model_loader
        if ldr is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Model loader not initialized in app state.",
            )
        champ = ldr.load()
        return {
            "status": "SUCCESS",
            "message": "Champion model hot-reloaded from MLflow Registry",
            "model_version": champ.version,
            "decision_threshold": champ.threshold,
        }

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"🚨 Unhandled Error on {request.method} {request.url.path}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing the request.",
                "detail": str(exc),
            },
        )

    return app


# Default ASGI app instance
app = create_app()
