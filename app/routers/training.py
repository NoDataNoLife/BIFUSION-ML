from fastapi import APIRouter
from app.schemas.training import TrainingRequest
from app.schemas.job import JobResponse, JobStatus
from app.workers.training_worker import run_training
from app.routers.augmentation import _map_celery_state

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_training(request: TrainingRequest):
    task = run_training.apply_async(
        args=[request.model_dump()],
        queue="training",
    )
    return JobResponse(job_id=task.id, status=JobStatus.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
def get_training_status(job_id: str):
    from app.core.celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    return JobResponse(
        job_id=job_id,
        status=_map_celery_state(task.state),
        result=task.result if task.state == "SUCCESS" else None,
        error=str(task.result) if task.state == "FAILURE" else None,
    )
