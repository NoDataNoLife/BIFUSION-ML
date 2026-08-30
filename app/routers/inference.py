from fastapi import APIRouter
from app.schemas.inference import InferenceRequest
from app.schemas.job import JobResponse, JobStatus
from app.workers.inference_worker import run_inference
from app.core.celery_app import celery_app
from app.routers.augmentation import _map_celery_state

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_inference(request: InferenceRequest):
    task = run_inference.apply_async(
        args=[request.model_dump()],
        queue="inference",
    )
    return JobResponse(job_id=task.id, status=JobStatus.PENDING, progress=0)


@router.get("/{job_id}", response_model=JobResponse)
def get_inference_status(job_id: str):
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
