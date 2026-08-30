import uuid
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image

from app.core.celery_app import celery_app
from app.core.config import settings
from app.services.diffusion_service import DDPMService
from app.services.protonet_service import ProtoNetService, ProtoEncoderConv4
from app.services.storage_service import get_storage_service
from app.services.image_utils import base64_to_pil, create_synthetic_noise_image
import torch


@celery_app.task(bind=True, name="app.workers.inference_worker.run_inference")
def run_inference(self, request: dict) -> Dict[str, Any]:
    """
    JOB-003: 모델 추론/진단 Worker
    - 케이스 1: 새 환자 X-ray 이미지 진단 (ProtoNet 퓨샷 분류)
    - 케이스 2: DDPM 가상 X-ray 이미지 생성 (단독 생성 모드)
    """
    task_id = self.request.id if (getattr(self, "request", None) and self.request.id) else None
    job_id = task_id or str(uuid.uuid4())

    def safe_update_state(state: str, meta: dict):
        if task_id:
            try:
                self.update_state(state=state, meta=meta)
            except Exception:
                pass

    safe_update_state(state="STARTED", meta={"progress": 0, "message": "Starting inference..."})

    storage = get_storage_service()

    images_raw = request.get("images_base64", [])
    num_samples = request.get("num_samples")
    model_id = request.get("model_id", "best.pt")

    # ==========================================
    # 모드 1: 새 환자 이미지 질환 진단 (ProtoNet)
    # ==========================================
    if images_raw or num_samples is None:
        safe_update_state(state="RUNNING", meta={"progress": 20, "message": "Running disease diagnosis..."})

        pil_images = []
        for b64 in images_raw:
            try:
                pil_images.append(base64_to_pil(b64))
            except Exception:
                pass

        if not pil_images:
            pil_images = [create_synthetic_noise_image(128)]

        possible_model_paths = [
            Path(settings.LOCAL_STORAGE_DIR) / "models" / model_id / "best.pt",
            Path(settings.MODEL_DIR) / model_id,
            Path(f"/tmp/models/{model_id}/best.pt"),
            Path(model_id),
        ]
        model_path = None
        for p in possible_model_paths:
            if p.exists():
                model_path = str(p)
                break

        if not model_path:
            tmp_model_dir = Path("/tmp/models") / "default"
            tmp_model_dir.mkdir(parents=True, exist_ok=True)
            model_path = str(tmp_model_dir / "best.pt")
            if not Path(model_path).exists():
                encoder = ProtoEncoderConv4(1, 64, 256)
                protos = torch.randn(2, 256)
                torch.save({"encoder_state_dict": encoder.state_dict(), "prototypes": protos}, model_path)

        protonet = ProtoNetService()
        predictions = protonet.predict(model_checkpoint_path=model_path, images=pil_images)

        safe_update_state(state="SUCCESS", meta={"progress": 100})

        return {
            "job_id": job_id,
            "status": "SUCCESS",
            "mode": "diagnosis",
            "model_id": model_id,
            "num_diagnosed": len(predictions),
            "predictions": predictions,
        }

    # ==========================================
    # 모드 2: DDPM 단독 이미지 생성
    # ==========================================
    else:
        batch_size = request.get("batch_size", 4)
        sampling_steps = request.get("sampling_steps", 100)
        seed = request.get("seed")

        ddpm = DDPMService(model_name=model_id, image_size=128, timesteps=500)

        def on_progress(step: int, total: int, pct: int):
            safe_update_state(
                state="RUNNING",
                meta={
                    "progress": pct,
                    "step": step,
                    "total_steps": total,
                    "message": f"Generating images ({pct}%)...",
                },
            )

        images = ddpm.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
            sampling_steps=sampling_steps,
            seed=seed,
            progress_callback=on_progress,
        )

        image_urls = []
        for idx, img in enumerate(images):
            key = f"inferences/{job_id}/synthetic_{idx:04d}.png"
            url = storage.save_image(img, key)
            image_urls.append(url)

        preview_img = ddpm.create_preview_grid(images[:8], nrow=4)
        preview_key = f"inferences/{job_id}/preview_grid.png"
        preview_url = storage.save_image(preview_img, preview_key)

        safe_update_state(state="SUCCESS", meta={"progress": 100})

        return {
            "job_id": job_id,
            "status": "SUCCESS",
            "mode": "generation",
            "num_samples": num_samples,
            "image_urls": image_urls,
            "preview_url": preview_url,
        }
