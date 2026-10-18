from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import Mock

import fakeredis
import pytest
from celery.exceptions import Ignore
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from app.main import app
from app.core import job_registry
from app.core.managed_task import ManagedTask
from app.services import job_service

client = TestClient(app)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr("app.main.settings.API_KEY", "secret-test-key")
    monkeypatch.setitem(client.headers, "X-API-Key", "secret-test-key")
    db = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(job_registry, "connection", lambda: db)
    return db


@pytest.mark.parametrize("kind", ["augmentation", "training", "inference"])
def test_missing_and_wrong_type_jobs_return_distinct_404(kind):
    for job_id in ["missing", "wrong-type"]:
        if job_id == "wrong-type":
            job_registry.register(job_id, "other")
        for method, path in [(client.get, f"/api/v1/{kind}/{job_id}"),
                             (client.post, f"/api/v1/{kind}/{job_id}/cancel")]:
            response = method(path)
            assert response.status_code == 404
            assert response.json()["detail"]["error_code"] == "JOB_NOT_FOUND"
    assert client.post(f"/api/v1/{kind}/missing/nonexistent").json() == {"detail": "Not Found"}


@pytest.mark.parametrize("kind", ["augmentation", "training", "inference"])
def test_cancel_is_idempotent_and_worker_skips(kind):
    job_registry.register("job", kind)
    assert client.get(f"/api/v1/{kind}/job").json()["status"] == "PENDING"
    for _ in range(2):
        response = client.post(f"/api/v1/{kind}/job/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "CANCELLED"
    with pytest.raises(Ignore):
        ManagedTask().before_start("job", [], {})
    assert client.get(f"/api/v1/{kind}/job").json()["status"] == "CANCELLED"


@pytest.mark.parametrize("state", ["RUNNING", "SUCCESS", "FAILED"])
def test_started_or_finished_cannot_cancel(state):
    job_registry.register("job", "training")
    assert job_registry.transition("job", "PENDING", state)
    response = client.post("/api/v1/training/job/cancel")
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "JOB_NOT_CANCELLABLE"


def test_worker_claim_and_terminal_hooks():
    task = ManagedTask()
    job_registry.register("ok", "training")
    task.before_start("ok", [], {})
    with pytest.raises(Ignore):
        task.before_start("ok", [], {})
    task.on_success({}, "ok", [], {})
    assert job_registry.lookup("ok")["state"] == "SUCCESS"
    job_registry.register("bad", "training")
    task.before_start("bad", [], {})
    task.on_failure(ValueError(), "bad", [], {}, None)
    assert job_registry.lookup("bad")["state"] == "FAILED"


def test_start_cancel_race_has_exactly_one_winner():
    for index in range(20):
        job_id = str(index)
        job_registry.register(job_id, "augmentation")
        barrier = Barrier(2)
        def compete(target):
            barrier.wait()
            return job_registry.transition(job_id, "PENDING", target)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(compete, ["RUNNING", "CANCELLED"]))
        assert sum(results) == 1


def test_submission_registers_before_publish():
    task = Mock()
    def publish(**kwargs):
        assert job_registry.lookup(kwargs["task_id"]) == {"kind": "training", "state": "PENDING"}
        assert kwargs["queue"] == "training"
    task.apply_async.side_effect = publish
    response = job_service.submit(task, "training", {"dataset_id": "dataset"})
    assert response.job_id == task.apply_async.call_args.kwargs["task_id"]


def test_failed_submission_cannot_later_start():
    task = Mock()
    task.apply_async.side_effect = RuntimeError("broker unavailable")
    with pytest.raises(Exception) as error:
        job_service.submit(task, "training", {})
    assert error.value.status_code == 503
    job_id = task.apply_async.call_args.kwargs["task_id"]
    assert job_registry.lookup(job_id)["state"] == "FAILED"
    with pytest.raises(Ignore):
        ManagedTask().before_start(job_id, [], {})


def test_redis_outage_is_not_job_not_found(monkeypatch):
    monkeypatch.setattr(job_registry, "lookup", Mock(side_effect=ConnectionError("offline")))
    for method, path in [(client.get, "/api/v1/training/job"),
                         (client.post, "/api/v1/training/job/cancel")]:
        response = method(path)
        assert response.status_code == 503
        assert response.json()["detail"]["error_code"] == "JOB_SERVICE_UNAVAILABLE"


@pytest.mark.parametrize("state,value,expected", [("SUCCESS", {"model_id": "m"}, "SUCCESS"),
                                                  ("FAILURE", ValueError("bad"), "FAILED"),
                                                  ("RUNNING", {"progress": 50}, "RUNNING")])
def test_result_status_mapping(monkeypatch, state, value, expected):
    job_registry.register("job", "training")
    job_registry.transition("job", "PENDING", "RUNNING")
    result = Mock()
    result._get_task_meta.return_value = {"status": state, "result": value}
    monkeypatch.setattr(job_service.celery_app, "AsyncResult", lambda _: result)
    response = client.get("/api/v1/training/job")
    assert response.status_code == 200
    assert response.json()["status"] == expected


def test_expired_result_is_not_unknown_job(monkeypatch):
    job_registry.register("job", "training")
    job_registry.transition("job", "PENDING", "SUCCESS")
    result = Mock()
    result._get_task_meta.return_value = {"status": "PENDING", "result": None}
    monkeypatch.setattr(job_service.celery_app, "AsyncResult", lambda _: result)
    response = client.get("/api/v1/training/job")
    assert response.status_code == 410
    assert response.json()["detail"]["error_code"] == "JOB_RESULT_EXPIRED"


def test_cancelled_task_delivery_never_executes_model():
    from app.core.celery_app import celery_app
    called = []
    @celery_app.task(base=ManagedTask, name="test.cancelled_delivery")
    def probe():
        called.append(True)
    job_registry.register("cancelled-delivery", "training")
    job_registry.transition("cancelled-delivery", "PENDING", "CANCELLED")
    result = probe.apply(task_id="cancelled-delivery")
    assert result.state == "IGNORED"
    assert called == []
