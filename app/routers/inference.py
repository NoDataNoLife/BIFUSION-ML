from fastapi import APIRouter
from app.schemas.inference import InferenceRequest
from app.schemas.job import JobResponse, JobStatus
from app.workers.inference_worker import run_inference
from app.routers.augmentation import _map_celery_state

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_inference(request: InferenceRequest):
    task = run_inference.apply_async(
        args=[request.model_dump()],
        queue="inference",
    )
    return JobResponse(job_id=task.id, status=JobStatus.PENDING)


@router.get("/{job_id}", response_model=JobResponse)
def get_inference_status(job_id: str):
    from app.core.celery_app import celery_app
    task = celery_app.AsyncResult(job_id)
    return JobResponse(
        job_id=job_id,
        status=_map_celery_state(task.state),
        result=task.result if task.state == "SUCCESS" else None,
        error=str(task.result) if task.state == "FAILURE" else None,
    )
