import io
import json
import uuid
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from app.core.celery_app import celery_app
from app.services.diffusion_service import DDPMService
from app.services.storage_service import get_storage_service
from app.services.image_utils import base64_to_pil, create_synthetic_noise_image


@celery_app.task(bind=True, name="app.workers.augmentation_worker.run_augmentation")
def run_augmentation(self, request: dict) -> Dict[str, Any]:
    """
    JOB-001: 데이터 증강 Worker
    1. Normal / Anomaly 원본 이미지 수신
    2. DDPM 모델을 통해 각 클래스당 N개의 합성 X-ray 이미지 생성
    3. 원본 + 합성 이미지를 결합하여 support_aug.npz 및 meta.json 생성
    4. S3 / 로컬 스토리지에 저장 및 URL 반환
    """
    task_id = self.request.id if (getattr(self, "request", None) and self.request.id) else None
    job_id = task_id or str(uuid.uuid4())

    def safe_update_state(state: str, meta: dict):
        if task_id:
            try:
                self.update_state(state=state, meta=meta)
            except Exception:
                pass

    safe_update_state(state="STARTED", meta={"progress": 0, "message": "Starting DDPM Augmentation..."})

    num_samples_per_class = request.get("num_samples_per_class", 10)
    sampling_steps = request.get("sampling_steps", 100)
    seed = request.get("seed", None)
    model_name = request.get("model_name", "best_model.pt")

    storage = get_storage_service()

    # 1. 입력 이미지 파싱
    normal_raw = request.get("normal_images_base64", [])
    anomaly_raw = request.get("anomaly_images_base64", [])

    normal_imgs: List[Image.Image] = []
    anomaly_imgs: List[Image.Image] = []

    for b64 in normal_raw:
        try:
            normal_imgs.append(base64_to_pil(b64).resize((128, 128)))
        except Exception:
            pass

    for b64 in anomaly_raw:
        try:
            anomaly_imgs.append(base64_to_pil(b64).resize((128, 128)))
        except Exception:
            pass

    if not normal_imgs:
        normal_imgs = [create_synthetic_noise_image(128) for _ in range(5)]
    if not anomaly_imgs:
        anomaly_imgs = [create_synthetic_noise_image(128) for _ in range(5)]

    # 2. DDPM 생성 서비스 초기화
    ddpm = DDPMService(model_name=model_name, image_size=128, timesteps=500)

    # 3. 진행률 콜백 정의
    def on_progress(step: int, total: int, pct: int):
        scaled_pct = int(10 + pct * 0.8)
        safe_update_state(
            state="RUNNING",
            meta={
                "progress": scaled_pct,
                "step": step,
                "total_steps": total,
                "message": f"Generating synthetic X-ray samples ({scaled_pct}%)...",
            },
        )

    # 4. 합성 이미지 생성 (Normal + Anomaly 증강)
    total_to_generate = num_samples_per_class * 2
    safe_update_state(state="RUNNING", meta={"progress": 10, "message": "Running diffusion sampling..."})

    synthetic_images = ddpm.generate_samples(
        num_samples=total_to_generate,
        batch_size=4,
        sampling_steps=sampling_steps,
        seed=seed,
        progress_callback=on_progress,
    )

    syn_normal = synthetic_images[:num_samples_per_class]
    syn_anomaly = synthetic_images[num_samples_per_class:]

    # 5. 전체 데이터셋 결합
    all_normal = normal_imgs + syn_normal
    all_anomaly = anomaly_imgs + syn_anomaly

    xs = []
    ys = []
    ids = []

    for idx, img in enumerate(all_normal):
        arr = np.array(img, dtype=np.float32) / 255.0
        xs.append(arr)
        ys.append(0)  # 0: Normal
        ids.append(f"normal_{idx:04d}")

    for idx, img in enumerate(all_anomaly):
        arr = np.array(img, dtype=np.float32) / 255.0
        xs.append(arr)
        ys.append(1)  # 1: Anomaly
        ids.append(f"anomaly_{idx:04d}")

    X_arr = np.stack(xs)[:, None, :, :]  # (N, 1, 128, 128)
    y_arr = np.array(ys, dtype=np.int64)
    ids_arr = np.array(ids, dtype=object)

    # 6. .npz 파일 생성 및 저장
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, x=X_arr, y=y_arr, ids=ids_arr)
    npz_key = f"datasets/{job_id}/support_aug.npz"
    dataset_url = storage.save_bytes(npz_buf.getvalue(), npz_key, content_type="application/octet-stream")

    # 7. 미리보기 그리드 이미지 생성 및 저장
    preview_img = ddpm.create_preview_grid(synthetic_images[:8], nrow=4)
    preview_key = f"datasets/{job_id}/preview_grid.png"
    preview_url = storage.save_image(preview_img, preview_key)

    # 8. 메타데이터 저장
    meta = {
        "job_type": "augmentation",
        "job_id": job_id,
        "dataset_id": job_id,
        "num_normal_total": len(all_normal),
        "num_anomaly_total": len(all_anomaly),
        "num_synthetic_generated": total_to_generate,
        "spec": {"size": [128, 128], "channels": 1, "normalize": "0-1"},
    }
    meta_key = f"datasets/{job_id}/meta.json"
    storage.save_bytes(json.dumps(meta, indent=2).encode("utf-8"), meta_key, content_type="application/json")

    safe_update_state(state="SUCCESS", meta={"progress": 100})

    return {
        "job_id": job_id,
        "status": "SUCCESS",
        "dataset_id": job_id,
        "dataset_url": dataset_url,
        "preview_url": preview_url,
        "num_normal": len(all_normal),
        "num_anomaly": len(all_anomaly),
        "total_samples": len(y_arr),
        "num_synthetic_generated": total_to_generate,
    }
