from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    APP_ENV: str = "local"  # local | prod

    # AI Model & Hardware
    DEVICE: str = "auto"  # auto | cuda | cpu | mps
    IMAGE_SIZE: int = 128
    CHANNELS: int = 1                 # 1=grayscale, 3=RGB (데이터셋/생성 출력 채널)

    # Medfusion (유일한 생성 백엔드)
    MEDFUSION_CODE_PATH: str = r"C:\Users\tjdgk\Desktop\My\PBL\medfusion_code"
    MEDFUSION_WEIGHTS_DIR: str = r"C:\Users\tjdgk\Desktop\My\PBL\medfusion_weights"
    MEDFUSION_DIFFUSION_CKPT: str = "runs/chest_diffusion/last.ckpt"
    MEDFUSION_VAE_CKPT: str = "runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt"
    MEDFUSION_LATENT_SHAPE: str = "8,32,32"
    MEDFUSION_GUIDANCE_SCALE: float = 4.0
    MEDFUSION_STEPS: int = 50

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
