from pydantic import BaseModel
from enum import Enum
from typing import Optional
from datetime import datetime


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: Optional[int] = None      # 0~100
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
