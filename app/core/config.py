from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Server
    APP_ENV: str = "local"  # local | prod

    # AI Model & Hardware
    DEVICE: str = "auto"  # auto | cuda | cpu | mps
    IMAGE_SIZE: int = 128
    CHANNELS: int = 3                 # 1=grayscale, 3=RGB. eye VAE는 RGB(색 보존)

    # 생성 로직이 담긴 bifusion 패키지(Bifusion_pip-version-Med) 경로.
    # 웹앱은 이 경로를 sys.path에 추가해 `bifusion`을 import한다(pip 설치 대신).
    BIFUSION_PACKAGE_PATH: str = r"C:\Users\tjdgk\Desktop\My\PBL\Bifusion_pip-version-Med"

    # Medfusion (유일한 생성 백엔드) — eye(fundus) VAE 단일 범용(RGB·회색조 모두 처리).
    # 실제 생성은 bifusion.generate.medfusion 이 담당.
    MEDFUSION_CODE_PATH: str = r"C:\Users\tjdgk\Desktop\My\PBL\medfusion_code"
    MEDFUSION_WEIGHTS_DIR: str = r"C:\Users\tjdgk\Desktop\My\PBL\medfusion_weights"
    MEDFUSION_DIFFUSION_CKPT: str = "runs/eye_diffusion/last.ckpt"
    MEDFUSION_VAE_CKPT: str = "runs/2022_12_12_224914_eye_vaegan/last_vae.ckpt"
    MEDFUSION_LATENT_SHAPE: str = "4,32,32"   # eye VAE latent (emb=4)
    MEDFUSION_GUIDANCE_SCALE: float = 4.0
    MEDFUSION_STEPS: int = 50

    # 증강: per-upload 파인튜닝(fidelity-first) 파라미터
    MEDFUSION_FINETUNE_STEPS: int = 400       # 업로드별 UNet 파인튜닝 스텝(VAE 동결)
    MEDFUSION_FINETUNE_LR: float = 1e-4
    MEDFUSION_STRENGTH: float = 0.3           # img2img 노이즈 비율(낮을수록 원본 밀착)

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
