import pytest
import fakeredis
from app.core import job_registry
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.workers import augmentation_worker, training_worker, inference_worker

client = TestClient(app)


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    monkeypatch.setattr("app.main.settings.API_KEY", "secret-test-key")
    monkeypatch.setitem(client.headers, "X-API-Key", "secret-test-key")
    redis = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(job_registry, "connection", lambda: redis)



def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_augmentation_run(monkeypatch):
    mock_task = MagicMock()
    mock_task.id = "mock-aug-job-123"
    monkeypatch.setattr(augmentation_worker.run_augmentation, "apply_async", lambda *args, **kwargs: mock_task)

    payload = {
        "num_samples_per_class": 2,
        "sampling_steps": 10,
        "normal_images_base64": [],
        "anomaly_images_base64": [],
    }
    response = client.post("/api/v1/augmentation/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert job_registry.lookup(data["job_id"])["state"] == "PENDING"
    assert data["status"] == "PENDING"


def test_api_training_run(monkeypatch):
    mock_task = MagicMock()
    mock_task.id = "mock-train-job-456"
    monkeypatch.setattr(training_worker.run_training, "apply_async", lambda *args, **kwargs: mock_task)

    payload = {
        "dataset_id": "test_dataset_123",
        "k_shot": 5,
        "epochs": 10,
    }
    response = client.post("/api/v1/training/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert job_registry.lookup(data["job_id"])["state"] == "PENDING"
    assert data["status"] == "PENDING"


def test_api_inference_run(monkeypatch):
    mock_task = MagicMock()
    mock_task.id = "mock-infer-job-789"
    monkeypatch.setattr(inference_worker.run_inference, "apply_async", lambda *args, **kwargs: mock_task)

    payload = {
        "model_id": "best.pt",
        "images_base64": [],
    }
    response = client.post("/api/v1/inference/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert job_registry.lookup(data["job_id"])["state"] == "PENDING"
    assert data["status"] == "PENDING"


def test_api_key_unauthorized_when_set(monkeypatch):
    monkeypatch.setattr("app.main.settings.API_KEY", "secret-test-key")
    monkeypatch.delitem(client.headers, "X-API-Key")
    # Missing key
    res = client.post("/api/v1/augmentation/run", json={})
    assert res.status_code == 401
    assert res.json()["detail"]["error_code"] == "UNAUTHORIZED"

    # Invalid key
    res2 = client.post("/api/v1/augmentation/run", json={}, headers={"X-API-Key": "wrong-key"})
    assert res2.status_code == 401
    assert res2.json()["detail"]["error_code"] == "UNAUTHORIZED"

    # Health check is always public even if API_KEY is set
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {"status": "ok"}


def test_api_key_authorized_when_set(monkeypatch):
    monkeypatch.setattr("app.main.settings.API_KEY", "secret-test-key")
    mock_task = MagicMock()
    mock_task.id = "mock-aug-job-auth"
    monkeypatch.setattr(augmentation_worker.run_augmentation, "apply_async", lambda *args, **kwargs: mock_task)

    payload = {
        "num_samples_per_class": 2,
        "sampling_steps": 10,
        "normal_images_base64": [],
        "anomaly_images_base64": [],
    }
    res = client.post("/api/v1/augmentation/run", json=payload, headers={"X-API-Key": "secret-test-key"})
    assert res.status_code == 200

@pytest.mark.parametrize("kind", ["augmentation", "training", "inference"])
@pytest.mark.parametrize("operation", ["run", "get", "cancel"])
@pytest.mark.parametrize("server_key,request_key,expected", [
    ("", "secret-test-key", 503),
    ("   ", "secret-test-key", 503),
    ("secret-test-key", None, 401),
    ("secret-test-key", "wrong", 401),
])
def test_all_job_routes_require_configured_key(monkeypatch, kind, operation, server_key, request_key, expected):
    monkeypatch.setattr("app.main.settings.API_KEY", server_key)
    monkeypatch.delitem(client.headers, "X-API-Key")
    headers = {"X-API-Key": request_key} if request_key is not None else {}
    if operation == "get":
        response = client.get(f"/api/v1/{kind}/missing", headers=headers)
    else:
        path = "run" if operation == "run" else "missing/cancel"
        response = client.post(f"/api/v1/{kind}/{path}", json={}, headers=headers)
    assert response.status_code == expected
    assert response.json()["detail"]["error_code"] == (
        "API_KEY_NOT_CONFIGURED" if expected == 503 else "UNAUTHORIZED"
    )
    assert client.get("/health").status_code == 200