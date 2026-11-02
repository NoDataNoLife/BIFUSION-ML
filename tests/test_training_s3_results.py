from pathlib import Path
from unittest.mock import Mock

import pytest

from app.core.config import settings
from app.services import storage_service
from app.workers import training_worker as worker


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "TRAINING_PUBLISH_S3", True)
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "test-results")
    monkeypatch.setattr(settings, "MODEL_S3_BUCKET", "")
    monkeypatch.setattr(settings, "S3_OUTPUT_PREFIX", "ml-results")
    dataset = tmp_path / "datasets/input/support_aug.npz"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"test dataset")
    monkeypatch.chdir(tmp_path)
    client = Mock()
    monkeypatch.setattr(storage_service.boto3, "client", lambda *a, **kw: client)
    trainer = Mock()
    def train(**kwargs):
        Path(kwargs["out_model_path"]).write_bytes(b"test checkpoint")
        return {"metrics": {"accuracy": 0.9}}
    trainer.train_few_shot.side_effect = train
    monkeypatch.setattr(worker, "ProtoNetService", lambda: trainer)
    monkeypatch.setattr(worker.run_training, "update_state", Mock())
    # Isolate the worker's temporary checkpoint directory as well.
    real_path = Path
    monkeypatch.setattr(worker, "Path", lambda p: (
        tmp_path / "temporary-models" if p == "/tmp/models" else real_path(p)
    ))
    return tmp_path, client, trainer


def run():
    worker.run_training.push_request(id="training-test")
    try:
        return worker.run_training.run({"dataset_id": "input", "epochs": 1})
    finally:
        worker.run_training.pop_request()


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_upload_copy_keeps_inference_model_and_returns_metadata(environment, monkeypatch, backend):
    root, client, _ = environment
    monkeypatch.setattr(settings, "STORAGE_BACKEND", backend)
    if backend == "s3":
        monkeypatch.setattr(settings, "TRAINING_PUBLISH_S3", False)
    uploaded = []
    client.upload_file.side_effect = lambda path, bucket, key, **kw: uploaded.append(
        (Path(path).read_bytes(), bucket, key)
    )
    result = run()
    local = root / "models/training-test/best.pt"
    assert local.read_bytes() == b"test checkpoint"
    assert result["model_url"] == str(local)
    assert result["artifacts"]["download"] == {
        "bucket": "test-results",
        "key": "ml-results/models/training-test/best.pt",
        "size_bytes": len(b"test checkpoint"),
    }
    assert uploaded == [(b"test checkpoint", "test-results",
                         "ml-results/models/training-test/best.pt")]
    assert result["metrics"] == {"accuracy": 0.9}
    assert result["status"] == "SUCCESS"
    client.generate_presigned_url.assert_not_called()


def test_upload_failure_preserves_local_model_and_fails_job(environment):
    root, client, _ = environment
    client.upload_file.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        run()
    assert (root / "models/training-test/best.pt").read_bytes() == b"test checkpoint"


def test_disabled_publication_is_local_only(environment, monkeypatch):
    _, client, _ = environment
    monkeypatch.setattr(settings, "TRAINING_PUBLISH_S3", False)
    result = run()
    assert result["artifacts"] is None
    assert Path(result["model_url"]).is_file()
    client.upload_file.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("AWS_S3_BUCKET_NAME", ""),
    ("S3_OUTPUT_PREFIX", "outputs"),
])
def test_invalid_publication_config_fails_before_training(environment, monkeypatch, field, value):
    _, client, trainer = environment
    monkeypatch.setattr(settings, field, value)
    with pytest.raises(ValueError):
        run()
    trainer.train_few_shot.assert_not_called()
    client.upload_file.assert_not_called()
