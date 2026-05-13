from fastapi import APIRouter
from app.schemas.augmentation import AugmentationRequest
from app.schemas.job import JobResponse, JobStatus
from app.workers.augmentation_worker import run_augmentation

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_augmentation(request: AugmentationRequest):
    task = run_augmentation.apply_async(
        args=[request.model_dump()],
        queue="augmentation",
    )
    return JobResponse(job_id=task.id, status=JobStatus.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
def get_augmentation_status(job_id: str):
    from app.core.celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    return JobResponse(
        job_id=job_id,
        status=_map_celery_state(task.state),
        result=task.result if task.state == "SUCCESS" else None,
        error=str(task.result) if task.state == "FAILURE" else None,
    )


def _map_celery_state(state: str) -> JobStatus:
    return {
        "PENDING": JobStatus.PENDING,
        "STARTED": JobStatus.RUNNING,
        "SUCCESS": JobStatus.SUCCESS,
        "FAILURE": JobStatus.FAILED,
        "REVOKED": JobStatus.CANCELLED,
    }.get(state, JobStatus.PENDING)
