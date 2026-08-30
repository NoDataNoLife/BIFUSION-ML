import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torch_ema import ExponentialMovingAverage

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


# ==========================================
# 싱글톤 모델 캐시 (워커 프로세스 메모리 상주)
# ==========================================
_DDPM_MODEL_CACHE: Dict[str, Tuple[UNet2DModel, DDPMScheduler, ExponentialMovingAverage]] = {}


def load_ddpm_pipeline(
    model_name: Optional[str] = None,
    image_size: Optional[int] = None,
    timesteps: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[UNet2DModel, DDPMScheduler, ExponentialMovingAverage, torch.device]:
    """
    DDPM UNet2DModel, Scheduler, EMA를 싱글톤으로 메모리에 로드
    """
    model_name = model_name or settings.DEFAULT_MODEL_NAME
    image_size = image_size or settings.IMAGE_SIZE
    timesteps = timesteps or settings.DEFAULT_TIMESTEPS
    target_device = device or get_device()

    cache_key = f"{model_name}_{image_size}_{timesteps}_{target_device}"

    if cache_key in _DDPM_MODEL_CACHE:
        model, noise_scheduler, ema = _DDPM_MODEL_CACHE[cache_key]
        return model, noise_scheduler, ema, target_device

    print(f"[INFO] Initializing DDPM Model (image_size={image_size}, timesteps={timesteps}, device={target_device})...")

    # 1. 모델 골격 생성 (NDNL-docker generate.py 스펙)
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(target_device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    # 2. 가중치 파일 로드
    loader = get_model_loader()
    try:
        model_path = loader.get_model_path(model_name)
        checkpoint = torch.load(str(model_path), map_location=target_device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            print(f"[INFO] Successfully loaded checkpoint from {model_path}")
        else:
            print(f"[WARN] Checkpoint format unexpected, loaded state directly or using initial weights.")
    except Exception as e:
        print(f"[WARN] Failed to load checkpoint file '{model_name}': {e}. Using initialized weights for serving.")

    model.eval()
    _DDPM_MODEL_CACHE[cache_key] = (model, noise_scheduler, ema)
    return model, noise_scheduler, ema, target_device
