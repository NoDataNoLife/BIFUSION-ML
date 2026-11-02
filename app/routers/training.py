from fastapi import APIRouter
from app.schemas.training import TrainingRequest
from app.schemas.job import JobResponse
from app.workers.training_worker import run_training
from app.services import job_service

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_training(request: TrainingRequest):
    return job_service.submit(run_training, "training", request.model_dump())


@router.get("/{job_id}", response_model=JobResponse)
def get_training_status(job_id: str):
    return job_service.status(job_id, "training")


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_training(job_id: str):
    return job_service.cancel(job_id, "training")
