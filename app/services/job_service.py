from uuid import uuid4
from fastapi import HTTPException
from redis.exceptions import RedisError
from app.core import job_registry
from app.core.celery_app import celery_app
from app.schemas.job import JobResponse, JobStatus


def unavailable():
    return HTTPException(503, detail={"error_code": "JOB_SERVICE_UNAVAILABLE",
                                     "message": "작업 저장소에 연결할 수 없습니다."})


def require_job(job_id, kind):
    try:
        record = job_registry.lookup(job_id)
    except RedisError as exc:
        raise unavailable() from exc
    if not record or record.get("kind") != kind:
        raise HTTPException(404, detail={"error_code": "JOB_NOT_FOUND",
                                         "message": "요청한 작업을 찾을 수 없습니다."})
    return record


def submit(task, kind, payload):
    job_id = str(uuid4())
    try:
        job_registry.register(job_id, kind)
    except RedisError as exc:
        raise unavailable() from exc
    try:
        task.apply_async(args=[payload], queue=kind, task_id=job_id)
    except Exception as exc:
        # A delivery whose submission failed must not later start as a fresh job.
        try:
            job_registry.transition(job_id, "PENDING", "FAILED")
        except RedisError:
            pass
        raise unavailable() from exc
    return JobResponse(job_id=job_id, status=JobStatus.PENDING, progress=0)


def status(job_id, kind):
    record = require_job(job_id, kind)
    state = record["state"]
    if state in ("PENDING", "CANCELLED"):
        return JobResponse(job_id=job_id, status=JobStatus(state))
    try:
        task = celery_app.AsyncResult(job_id)
        meta = task._get_task_meta()
    except RedisError as exc:
        raise unavailable() from exc
    celery_state = meta["status"]
    value = meta.get("result")
    # A worker may write SUCCESS before its lifecycle hook updates the registry.
    if celery_state == "SUCCESS":
        return JobResponse(job_id=job_id, status=JobStatus.SUCCESS, progress=100, result=value)
    if celery_state == "FAILURE":
        return JobResponse(job_id=job_id, status=JobStatus.FAILED, error=str(value))
    if state == "FAILED":
        return JobResponse(job_id=job_id, status=JobStatus.FAILED, error="작업 실행 또는 접수에 실패했습니다.")
    if state == "SUCCESS":
        raise HTTPException(410, detail={"error_code": "JOB_RESULT_EXPIRED",
                                         "message": "작업 결과 보관 기간이 만료되었습니다."})
    progress = value.get("progress", 0) if isinstance(value, dict) else 0
    return JobResponse(job_id=job_id, status=JobStatus.RUNNING, progress=progress)


def cancel(job_id, kind):
    record = require_job(job_id, kind)
    try:
        if record["state"] == "CANCELLED" or job_registry.transition(job_id, "PENDING", "CANCELLED"):
            return JobResponse(job_id=job_id, status=JobStatus.CANCELLED)
        # Another cancellation may have won between the first read and transition.
        if require_job(job_id, kind)["state"] == "CANCELLED":
            return JobResponse(job_id=job_id, status=JobStatus.CANCELLED)
    except RedisError as exc:
        raise unavailable() from exc
    raise HTTPException(409, detail={"error_code": "JOB_NOT_CANCELLABLE",
                                     "message": "이미 시작되었거나 종료된 작업은 취소할 수 없습니다."})
