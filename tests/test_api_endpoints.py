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
