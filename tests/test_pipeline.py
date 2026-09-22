import numpy as np
from pathlib import Path
from PIL import Image

from app.services.protonet_service import ProtoNetService
from app.services.storage_service import LocalStorageService
from app.services.image_utils import base64_to_pil, pil_to_base64, create_synthetic_noise_image


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
