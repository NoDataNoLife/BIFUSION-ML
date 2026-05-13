from abc import ABC, abstractmethod
from pathlib import Path
from app.core.config import settings


class ModelLoader(ABC):
    @abstractmethod
    def load(self, model_name: str):
        pass


class LocalModelLoader(ModelLoader):
    def load(self, model_name: str):
        model_path = Path(settings.MODEL_DIR) / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        # TODO: 실제 모델 로딩 로직으로 교체 (torch.load, etc.)
        return str(model_path)


class S3ModelLoader(ModelLoader):
    def __init__(self):
        import boto3
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.MODEL_S3_BUCKET
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, model_name: str):
        cache_path = self.cache_dir / model_name
        if not cache_path.exists():
            self.s3.download_file(self.bucket, model_name, str(cache_path))
        # TODO: 실제 모델 로딩 로직으로 교체
        return str(cache_path)


def get_model_loader() -> ModelLoader:
    if settings.MODEL_LOADER == "s3":
        return S3ModelLoader()
    return LocalModelLoader()
