from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    APP_ENV: str = "local"  # local | prod

    # AI Model & Hardware
    DEVICE: str = "auto"  # auto | cuda | cpu | mps
    IMAGE_SIZE: int = 128
    DEFAULT_TIMESTEPS: int = 500
    DEFAULT_MODEL_NAME: str = "best_model.pt"

    # Model Loader
    MODEL_LOADER: str = "local"  # local | s3
    MODEL_DIR: str = "models"
    MODEL_CACHE_DIR: str = "/tmp/models"

    # Storage Backend (local | s3)
    STORAGE_BACKEND: str = "local"  # local | s3
    LOCAL_STORAGE_DIR: str = "storage"
    S3_OUTPUT_PREFIX: str = "outputs"

    # S3 (프로덕션 전환 또는 STORAGE_BACKEND=s3 사용 시)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    MODEL_S3_BUCKET: str = ""
    AWS_S3_BUCKET_NAME: str = ""

    # Redis (Celery broker)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"

    # DB (job 상태 저장)
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/bifusion"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
