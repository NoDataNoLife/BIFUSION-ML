from fastapi import APIRouter
from app.schemas.augmentation import AugmentationRequest
from app.schemas.job import JobResponse
from app.workers.augmentation_worker import run_augmentation
from app.services import job_service

router = APIRouter()


@router.post("/run", response_model=JobResponse)
def start_augmentation(request: AugmentationRequest):
    return job_service.submit(run_augmentation, "augmentation", request.model_dump())


@router.get("/{job_id}", response_model=JobResponse)
def get_augmentation_status(job_id: str):
    return job_service.status(job_id, "augmentation")


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_augmentation(job_id: str):
    return job_service.cancel(job_id, "augmentation")
