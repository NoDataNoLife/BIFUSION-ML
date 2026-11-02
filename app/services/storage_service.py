import io
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional
import boto3
from PIL import Image

from app.core.config import settings


class StorageService(ABC):
    @abstractmethod
    def save_bytes(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        """바이트 데이터를 저장하고 접근 가능한 URL/경로를 반환"""
        pass

    @abstractmethod
    def save_image(self, image: Image.Image, key: str, format: str = "PNG") -> str:
        """PIL 이미지를 저장하고 접근 가능한 URL/경로를 반환"""
        pass

    @abstractmethod
    def save_file(self, local_path: Union[str, Path], key: str, content_type: Optional[str] = None) -> str:
        """로컬 파일을 스토리지에 저장하고 URL/경로를 반환"""
        pass

    @abstractmethod
    def get_url(self, key: str) -> str:
        """스토리지 키에 대한 접근 URL 반환"""
        pass


class LocalStorageService(StorageService):
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or settings.LOCAL_STORAGE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        target_path = self.base_dir / key
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(data)
        return str(target_path)

    def save_image(self, image: Image.Image, key: str, format: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=format)
        return self.save_bytes(buf.getvalue(), key, content_type=f"image/{format.lower()}")

    def save_file(self, local_path: Union[str, Path], key: str, content_type: Optional[str] = None) -> str:
        src = Path(local_path)
        target = self.base_dir / key
        target.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != target.resolve():
            import shutil
            shutil.copy2(src, target)
        return str(target)

    def get_url(self, key: str) -> str:
        target_path = self.base_dir / key
        return str(target_path.resolve())


class S3StorageService(StorageService):
    def __init__(self):
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            region_name=settings.AWS_REGION,
        )
        self.bucket = settings.AWS_S3_BUCKET_NAME or settings.MODEL_S3_BUCKET
        self.prefix = settings.S3_OUTPUT_PREFIX

    def _get_s3_key(self, key: str) -> str:
        clean_key = key.lstrip("/")
        if self.prefix and not clean_key.startswith(self.prefix):
            return f"{self.prefix}/{clean_key}"
        return clean_key

    def save_bytes(self, data: bytes, key: str, content_type: str = "image/png") -> str:
        s3_key = self._get_s3_key(key)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        return self.get_url(s3_key)

    def save_image(self, image: Image.Image, key: str, format: str = "PNG") -> str:
        buf = io.BytesIO()
        image.save(buf, format=format)
        return self.save_bytes(buf.getvalue(), key, content_type=f"image/{format.lower()}")

    def save_file(self, local_path: Union[str, Path], key: str, content_type: Optional[str] = None) -> str:
        s3_key = self._get_s3_key(key)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.s3.upload_file(str(local_path), self.bucket, s3_key, ExtraArgs=extra_args or None)
        return self.get_url(s3_key)

    def upload_artifact(self, local_path: Union[str, Path], key: str, content_type: str) -> dict:
        """Upload a local result and return durable metadata for BE persistence."""
        path = Path(local_path)
        s3_key = self._get_s3_key(key)
        size_bytes = path.stat().st_size
        self.s3.upload_file(
            str(path), self.bucket, s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        return {"bucket": self.bucket, "key": s3_key, "size_bytes": size_bytes}

    def get_url(self, key: str) -> str:
        s3_key = self._get_s3_key(key)
        try:
            # Presigned URL (1시간 유효) 생성
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=3600,
            )
            return url
        except Exception:
            return f"https://{self.bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"


def get_storage_service() -> StorageService:
    if settings.STORAGE_BACKEND.lower() == "s3" and (settings.AWS_S3_BUCKET_NAME or settings.MODEL_S3_BUCKET):
        return S3StorageService()
    return LocalStorageService()
