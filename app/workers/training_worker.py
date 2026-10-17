import os
import uuid
from pathlib import Path
from typing import Dict, Any

from app.core.celery_app import celery_app
from app.core.managed_task import ManagedTask
from app.core.config import settings
from app.services.protonet_service import ProtoNetService
from app.services.storage_service import get_storage_service


@celery_app.task(bind=True, base=ManagedTask, name="app.workers.training_worker.run_training")
def run_training(self, request: dict) -> Dict[str, Any]:
    """
    JOB-002: ProtoNet 퓨샷 모델 학습 Worker
    1. 1단계 증강 데이터셋(.npz) 로드
    2. ProtoEncoderConv4 기반 ProtoNet 퓨샷 학습
    3. 에포크별 정확도/AUROC 측정
    4. 최적 모델 가중치(best.pt) 저장 및 성능 성적표 반환
    """
    task_id = self.request.id if (getattr(self, "request", None) and self.request.id) else None
    job_id = task_id or str(uuid.uuid4())

    def safe_update_state(state: str, meta: dict):
        if task_id:
            try:
                self.update_state(state=state, meta=meta)
            except Exception:
                pass

    safe_update_state(state="STARTED", meta={"progress": 0, "message": "Starting Few-Shot ProtoNet training..."})

    dataset_id = request.get("dataset_id")
    k_shot = request.get("k_shot", 5)
    epochs = request.get("epochs", 30)
    q_query = request.get("q_query", 5)
    episodes_per_epoch = request.get("episodes_per_epoch", 10)
    seed = request.get("seed", 42)

    storage = get_storage_service()

    # 1. 데이터셋 파일 위치 탐색
    possible_paths = [
        Path(settings.LOCAL_STORAGE_DIR) / "datasets" / dataset_id / "support_aug.npz",
        Path(dataset_id),
        Path(f"datasets/{dataset_id}/support_aug.npz"),
    ]
    npz_path = None
    for p in possible_paths:
        if p.exists():
            npz_path = str(p)
            break

    if not npz_path:
        # 데이터셋 미발견 시 랜덤 노이즈로 학습해 "성공"으로 위장하지 않도록 명시적 실패
        raise FileNotFoundError(
            f"Dataset not found for dataset_id={dataset_id}. "
            f"Searched: {[str(p) for p in possible_paths]}"
        )

    # 2. 임시 모델 저장 경로
    out_model_dir = Path("/tmp/models") / job_id
    out_model_dir.mkdir(parents=True, exist_ok=True)
    out_model_path = str(out_model_dir / "best.pt")

    # 3. 진행률 콜백
    def on_progress(epoch: int, total_epochs: int, pct: int):
        safe_update_state(
            state="RUNNING",
            meta={
                "progress": pct,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "message": f"Training ProtoNet Epoch {epoch}/{total_epochs} ({pct}%)...",
            },
        )

    # 4. 학습 실행
    service = ProtoNetService()
    train_result = service.train_few_shot(
        npz_dataset_path=npz_path,
        k_shot=k_shot,
        epochs=epochs,
        out_model_path=out_model_path,
        progress_callback=on_progress,
        q_query=q_query,
        episodes_per_epoch=episodes_per_epoch,
        seed=seed,
    )

    # 5. 모델 가중치를 StorageService에 저장
    model_key = f"models/{job_id}/best.pt"
    model_url = storage.save_file(out_model_path, model_key, content_type="application/octet-stream")


    return {
        "job_id": job_id,
        "status": "SUCCESS",
        "model_id": job_id,
        "model_url": model_url,
        "dataset_id": dataset_id,
        "k_shot": k_shot,
        "metrics": train_result.get("metrics", {}),
    }
