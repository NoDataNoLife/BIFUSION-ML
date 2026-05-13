from fastapi import FastAPI
from app.routers import augmentation, training, inference
from app.core.config import settings

app = FastAPI(
    title="BIFUSION ML Server",
    version="0.1.0",
)

app.include_router(augmentation.router, prefix="/api/v1/augmentation", tags=["augmentation"])
app.include_router(training.router, prefix="/api/v1/training", tags=["training"])
app.include_router(inference.router, prefix="/api/v1/inference", tags=["inference"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
