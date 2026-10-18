import secrets
from fastapi import FastAPI, Depends, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.routers import augmentation, training, inference
from app.core.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)):
    """
    X-API-Key 헤더 검증.
    - 서버 키가 비어 있으면 작업 API는 503으로 차단합니다.
    - settings.API_KEY가 설정되어 있으면 X-API-Key 헤더가 일치해야 함.
    - 불일치 또는 누락 시 401 UNAUTHORIZED 에러 반환.
    """
    if not settings.API_KEY.strip():
        raise HTTPException(
            status_code=503,
            detail={"error_code": "API_KEY_NOT_CONFIGURED", "message": "API 인증 설정이 준비되지 않았습니다."},
        )
    if not api_key or not secrets.compare_digest(api_key.encode("utf-8"), settings.API_KEY.encode("utf-8")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error_code": "UNAUTHORIZED", "message": "유효하지 않거나 누락된 API 키입니다."},
        )


app = FastAPI(
    title="BIFUSION ML Server",
    version="0.1.0",
)

app.include_router(
    augmentation.router,
    prefix="/api/v1/augmentation",
    tags=["augmentation"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    training.router,
    prefix="/api/v1/training",
    tags=["training"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    inference.router,
    prefix="/api/v1/inference",
    tags=["inference"],
    dependencies=[Depends(verify_api_key)],
)


@app.get("/health")
def health_check():
    return {"status": "ok"}
