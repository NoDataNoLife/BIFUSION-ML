from fastapi import APIRouter
from app.schemas.augmentation import AugmentationRequest
from app.schemas.job import JobResponse, JobStatus
from app.workers.augmentation_worker import run_augmentation
from app.core.celery_app import celery_app

router = APIRouter()


def _map_celery_state(state: str) -> JobStatus:
    return {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.STARTED,
        "RUNNING": JobStatus.RUNNING,
        "SUCCESS": JobStatus.SUCCESS,
        "FAILURE": JobStatus.FAILED,
        "REVOKED": JobStatus.CANCELLED,
    }.get(state, JobStatus.PENDING)


@router.post("/run", response_model=JobResponse)
def start_augmentation(request: AugmentationRequest):
    task = run_augmentation.apply_async(
        args=[request.model_dump()],
        queue="augmentation",
    )
    return JobResponse(job_id=task.id, status=JobStatus.PENDING, progress=0)


@router.get("/{job_id}", response_model=JobResponse)
def get_augmentation_status(job_id: str):
    task = celery_app.AsyncResult(job_id)
    state = task.state
    status = _map_celery_state(state)

    progress = 0
    result = None
    error = None

    if state in ("STARTED", "RUNNING"):
        if isinstance(task.info, dict):
            progress = task.info.get("progress", 0)
    elif state == "SUCCESS":
        progress = 100
        result = task.result
    elif state == "FAILURE":
        error = str(task.result)

    return JobResponse(
        job_id=job_id,
        status=status,
        progress=progress,
        result=result,
        error=error,
    )
