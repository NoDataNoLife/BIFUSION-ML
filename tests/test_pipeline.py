import os
import shutil
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.core.config import settings
from app.services.diffusion_service import DDPMService
from app.services.protonet_service import ProtoNetService, ProtoEncoderConv4
from app.services.storage_service import LocalStorageService
from app.services.image_utils import base64_to_pil, pil_to_base64, create_synthetic_noise_image
from scripts.create_dummy_checkpoint import create_dummy_checkpoint


@pytest.fixture(scope="session", autouse=True)
def setup_dummy_model():
    """테스트 실행 전 더미 모델 생성"""
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        create_dummy_checkpoint(str(model_path), image_size=128)
    yield


def test_storage_service(tmp_path):
    storage = LocalStorageService(base_dir=str(tmp_path))
    img = Image.new("L", (128, 128), color=128)
    url = storage.save_image(img, "test/sample.png")
    assert Path(url).exists()
    assert (tmp_path / "test/sample.png").exists()


def test_image_utils():
    img = create_synthetic_noise_image(128)
    assert img.size == (128, 128)
    b64 = pil_to_base64(img)
    recovered = base64_to_pil(b64)
    assert recovered.size == (128, 128)


def test_diffusion_service_generation():
    service = DDPMService(model_name="best_model.pt", image_size=128, timesteps=50)
    progresses = []

    def on_progress(step, total, pct):
        progresses.append(pct)

    samples = service.generate_samples(
        num_samples=2,
        batch_size=2,
        sampling_steps=5,  # 테스트 속도를 위해 5스텝만
        seed=42,
        progress_callback=on_progress,
    )

    assert len(samples) == 2
    assert samples[0].size == (128, 128)
    assert len(progresses) > 0
    assert progresses[-1] == 100

    grid = service.create_preview_grid(samples, nrow=2)
    assert grid.size[0] > 0 and grid.size[1] > 0


def test_protonet_service(tmp_path):
    # 1. 가상 .npz 데이터셋 생성
    npz_path = str(tmp_path / "support_aug.npz")
    X = np.random.rand(20, 1, 128, 128).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    np.savez_compressed(npz_path, x=X, y=y)

    out_model = str(tmp_path / "best.pt")
    service = ProtoNetService()

    # 2. 퓨샷 학습 (빠른 테스트를 위해 2 에포크만)
    res = service.train_few_shot(
        npz_dataset_path=npz_path,
        k_shot=3,
        epochs=2,
        out_model_path=out_model,
    )

    assert "metrics" in res
    assert Path(out_model).exists()

    # 3. 추론/진단 테스트
    test_imgs = [create_synthetic_noise_image(128) for _ in range(3)]
    preds = service.predict(model_checkpoint_path=out_model, images=test_imgs)

    assert len(preds) == 3
    assert preds[0]["predicted_class"] in ("NORMAL", "ANOMALY")
    assert 0.0 <= preds[0]["confidence"] <= 1.0
