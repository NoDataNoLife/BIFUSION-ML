from fastapi import APIRouter
from app.schemas.inference import InferenceRequest
from app.schemas.job import JobResponse
from app.workers.inference_worker import run_inference
from app.services import job_service

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_inference(request: InferenceRequest):
    return job_service.submit(run_inference, "inference", request.model_dump())


@router.get("/{job_id}", response_model=JobResponse)
def get_inference_status(job_id: str):
    return job_service.status(job_id, "inference")


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_inference(job_id: str):
    return job_service.cancel(job_id, "inference")
