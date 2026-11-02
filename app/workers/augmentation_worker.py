import io
import json
import uuid
from typing import Dict, Any, List
import numpy as np
from PIL import Image

from app.core.celery_app import celery_app
from app.core.managed_task import ManagedTask
from app.core.config import settings
from app.services.medfusion_service import MedfusionService
from app.services.storage_service import LocalStorageService, S3StorageService
from app.services.image_utils import base64_to_pil


@celery_app.task(bind=True, base=ManagedTask, name="app.workers.augmentation_worker.run_augmentation")
def run_augmentation(self, request: dict) -> Dict[str, Any]:
    """
    JOB-001: 데이터 증강 Worker (fidelity-first)
    1. Normal / Anomaly 원본 이미지 수신
    2. 클래스별로 업로드에 Medfusion(eye VAE) UNet을 파인튜닝한 뒤 img2img로 각 클래스당 N개 생성
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

    safe_update_state(state="STARTED", meta={"progress": 0, "message": "Starting Medfusion augmentation..."})

    num_samples_per_class = request.get("num_samples_per_class", 10)
    seed = request.get("seed", None)
    strength = request.get("strength")              # None=서버 기본(MEDFUSION_STRENGTH)
    finetune_steps = request.get("finetune_steps")  # None=서버 기본(MEDFUSION_FINETUNE_STEPS)
    steps = request.get("steps")                    # None=서버 기본(MEDFUSION_STEPS)

    # Training reads local files even when results are also published to S3.
    storage = LocalStorageService()
    publish_s3 = settings.AUGMENTATION_PUBLISH_S3 or settings.STORAGE_BACKEND.lower() == "s3"
    if publish_s3 and not (settings.AWS_S3_BUCKET_NAME or settings.MODEL_S3_BUCKET):
        raise ValueError("S3 publication requires AWS_S3_BUCKET_NAME or MODEL_S3_BUCKET")
    publisher = S3StorageService() if publish_s3 else None

    # 1. 입력 이미지 파싱
    normal_raw = request.get("normal_images_base64", [])
    anomaly_raw = request.get("anomaly_images_base64", [])

    ch = int(settings.CHANNELS)              # 1=grayscale, 3=RGB
    pil_mode = "L" if ch == 1 else "RGB"

    normal_imgs: List[Image.Image] = []
    anomaly_imgs: List[Image.Image] = []

    for b64 in normal_raw:
        try:
            normal_imgs.append(base64_to_pil(b64).convert(pil_mode).resize((128, 128)))
        except Exception:
            pass

    for b64 in anomaly_raw:
        try:
            anomaly_imgs.append(base64_to_pil(b64).convert(pil_mode).resize((128, 128)))
        except Exception:
            pass

    # 2. 빈 클래스 방어 (fidelity-first: 원본 없으면 증강 불가 → 명시적 실패)
    if not normal_imgs:
        raise ValueError(
            f"No valid Normal images to augment (received {len(normal_raw)} entries, "
            f"all failed to decode)."
        )
    if not anomaly_imgs:
        raise ValueError(
            f"No valid Anomaly images to augment (received {len(anomaly_raw)} entries, "
            f"all failed to decode)."
        )

    # 3. 생성 서비스 (Medfusion: eye VAE + per-upload UNet 파인튜닝)
    generator = MedfusionService(image_size=128)

    def make_cb(lo: int, hi: int, label: str):
        def cb(step: int, total: int, pct: int):
            scaled = int(lo + (hi - lo) * pct / 100)
            safe_update_state(state="RUNNING", meta={
                "progress": scaled, "step": step, "total_steps": total,
                "message": f"Augmenting {label} (finetune+generate) {scaled}%...",
            })
        return cb

    safe_update_state(state="RUNNING", meta={"progress": 10, "message": "Fine-tuning on uploads & generating..."})

    # 4. 클래스별 per-upload 파인튜닝 + img2img 증강 (각 클래스는 자기 업로드로 특화)
    syn_normal = generator.augment_with_finetune(
        normal_imgs, num_outputs=num_samples_per_class,
        strength=strength, finetune_steps=finetune_steps, steps=steps, seed=seed,
        progress_callback=make_cb(10, 50, "Normal"),
    )
    syn_anomaly = generator.augment_with_finetune(
        anomaly_imgs, num_outputs=num_samples_per_class,
        strength=strength, finetune_steps=finetune_steps, steps=steps, seed=seed,
        progress_callback=make_cb(50, 90, "Anomaly"),
    )
    synthetic_images = syn_normal + syn_anomaly
    total_to_generate = len(synthetic_images)

    # 5. 전체 데이터셋 결합
    all_normal = normal_imgs + syn_normal
    all_anomaly = anomaly_imgs + syn_anomaly

    xs = []
    ys = []
    ids = []

    for idx, img in enumerate(all_normal):
        arr = np.array(img.convert(pil_mode), dtype=np.float32) / 255.0
        xs.append(arr)
        ys.append(0)  # 0: Normal
        ids.append(f"normal_{idx:04d}")

    for idx, img in enumerate(all_anomaly):
        arr = np.array(img.convert(pil_mode), dtype=np.float32) / 255.0
        xs.append(arr)
        ys.append(1)  # 1: Anomaly
        ids.append(f"anomaly_{idx:04d}")

    X_arr = np.stack(xs)
    if ch == 1:
        X_arr = X_arr[:, None, :, :]          # (N, 1, 128, 128)
    else:
        X_arr = np.transpose(X_arr, (0, 3, 1, 2))  # (N, 3, 128, 128)
    y_arr = np.array(ys, dtype=np.int64)
    ids_arr = np.array(ids, dtype=object)

    # 6. .npz 파일 생성 및 저장
    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, x=X_arr, y=y_arr, ids=ids_arr)
    npz_key = f"datasets/{job_id}/support_aug.npz"
    dataset_url = storage.save_bytes(npz_buf.getvalue(), npz_key, content_type="application/octet-stream")

    # 7. 미리보기 그리드 이미지 생성 및 저장
    preview_img = generator.create_preview_grid(synthetic_images[:8], nrow=4)
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
        "spec": {"size": [128, 128], "channels": ch, "normalize": "0-1"},
    }
    meta_key = f"datasets/{job_id}/meta.json"
    storage.save_bytes(json.dumps(meta, indent=2).encode("utf-8"), meta_key, content_type="application/json")

    # Keep local data; publish only download and preview artifacts.
    artifacts = None
    if publisher is not None:
        artifacts = {
            "download": publisher.upload_artifact(dataset_url, npz_key, "application/octet-stream"),
            "preview": publisher.upload_artifact(preview_url, preview_key, "image/png"),
        }
        # Preserve existing URL fields; BE should persist artifacts instead.
        dataset_url = publisher.get_url(artifacts["download"]["key"])
        preview_url = publisher.get_url(artifacts["preview"]["key"])


    return {
        "job_id": job_id,
        "status": "SUCCESS",
        "dataset_id": job_id,
        "artifacts": artifacts,
        "dataset_url": dataset_url,
        "preview_url": preview_url,
        "num_normal": len(all_normal),
        "num_anomaly": len(all_anomaly),
        "total_samples": len(y_arr),
        "num_synthetic_generated": total_to_generate,
    }
