import os
from abc import ABC, abstractmethod
from pathlib import Path
import torch

from app.core.config import settings


def get_device() -> torch.device:
    """하드웨어 환경에 맞는 최적의 PyTorch Device 반환"""
    cfg_device = settings.DEVICE.lower()
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg_device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if cfg_device == "cpu":
        return torch.device("cpu")

    # auto 감지
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ==========================================
# 모델/가중치 로더 (로컬 · S3) — 체크포인트 파일 경로 해석용 범용 인프라
# ==========================================
class ModelLoader(ABC):
    @abstractmethod
    def get_model_path(self, model_name: str) -> Path:
        pass


class LocalModelLoader(ModelLoader):
    def get_model_path(self, model_name: str) -> Path:
        model_path = Path(settings.MODEL_DIR) / model_name
        if not model_path.exists():
            # fallback: models 폴더 내에 없을 경우 상위 폴더 등 탐색
            if Path(model_name).exists():
                return Path(model_name)
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path.resolve()}")
        return model_path


class S3ModelLoader(ModelLoader):
    def __init__(self):
        import boto3
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.MODEL_S3_BUCKET or settings.AWS_S3_BUCKET_NAME
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_name: str) -> Path:
        cache_path = self.cache_dir / model_name
        if not cache_path.exists():
            print(f"[INFO] Downloading model {model_name} from S3 bucket {self.bucket}...")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, model_name, str(cache_path))
            print(f"[INFO] Model downloaded to {cache_path}")
        return cache_path


def get_model_loader() -> ModelLoader:
    if settings.MODEL_LOADER == "s3":
        return S3ModelLoader()
    return LocalModelLoader()
