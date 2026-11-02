from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from botocore.exceptions import ClientError

from app.core.config import settings
from app.services import storage_service
from app.services.image_utils import pil_to_base64
from app.workers import augmentation_worker as worker


@pytest.fixture
def environment(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "LOCAL_STORAGE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "AUGMENTATION_PUBLISH_S3", True)
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "test-results")
    monkeypatch.setattr(settings, "MODEL_S3_BUCKET", "")
    monkeypatch.setattr(settings, "S3_OUTPUT_PREFIX", "outputs")
    monkeypatch.setattr(settings, "CHANNELS", 3)
    client = Mock()
    client.generate_presigned_url.return_value = "https://example.invalid/temporary"
    monkeypatch.setattr(storage_service.boto3, "client", lambda *a, **kw: client)
    generator = Mock()
    generator.augment_with_finetune.return_value = [Image.new("RGB", (128, 128), "red")]
    generator.create_preview_grid.return_value = Image.new("RGB", (128, 128), "red")
    monkeypatch.setattr(worker, "MedfusionService", lambda **kw: generator)
    states = Mock()
    monkeypatch.setattr(worker.run_augmentation, "update_state", states)
    encoded = pil_to_base64(Image.new("RGB", (128, 128), "blue"))
    request = {"normal_images_base64": [encoded], "anomaly_images_base64": [encoded],
               "num_samples_per_class": 1}
    return tmp_path, client, generator, states, request


def run(request):
    worker.run_augmentation.push_request(id="test-job")
    try:
        return worker.run_augmentation.run(request)
    finally:
        worker.run_augmentation.pop_request()


@pytest.mark.parametrize("backend", ["local", "s3"])
def test_publish_preserves_training_data_and_returns_real_file_metadata(environment, monkeypatch, backend):
    root, client, _, _, request = environment
    monkeypatch.setattr(settings, "STORAGE_BACKEND", backend)
    if backend == "s3":
        monkeypatch.setattr(settings, "AUGMENTATION_PUBLISH_S3", False)
    uploaded = {}
    def capture(path, bucket, key, ExtraArgs):
        uploaded[key] = (Path(path).read_bytes(), ExtraArgs["ContentType"])
    client.upload_file.side_effect = capture
    result = run(request)
    folder = root / "datasets/test-job"
    with np.load(folder / "support_aug.npz") as data:
        assert data["x"].shape == (4, 3, 128, 128)
        assert data["y"].tolist() == [0, 0, 1, 1]
    assert (folder / "meta.json").exists()
    assert len(uploaded) == 2
    for kind, filename, mime in [("download", "support_aug.npz", "application/octet-stream"),
                                  ("preview", "preview_grid.png", "image/png")]:
        key = f"outputs/datasets/test-job/{filename}"
        assert result["artifacts"][kind] == {
            "bucket": "test-results", "key": key,
            "size_bytes": (folder / filename).stat().st_size,
        }
        assert uploaded[key] == ((folder / filename).read_bytes(), mime)
    assert result["status"] == "SUCCESS"
    assert result["preview_url"] == "https://example.invalid/temporary"


def test_local_only_does_not_upload(environment, monkeypatch):
    root, client, _, _, request = environment
    monkeypatch.setattr(settings, "AUGMENTATION_PUBLISH_S3", False)
    result = run(request)
    assert result["artifacts"] is None
    assert Path(result["dataset_url"]).exists()
    client.upload_file.assert_not_called()


def test_missing_bucket_fails_before_generation(environment, monkeypatch):
    _, client, generator, _, request = environment
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "")
    with pytest.raises(ValueError, match="S3 publication requires"):
        run(request)
    generator.augment_with_finetune.assert_not_called()
    client.upload_file.assert_not_called()


def test_partial_upload_failure_keeps_local_files_and_never_reports_success(environment):
    root, client, _, states, request = environment
    client.upload_file.side_effect = [None, ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject")]
    with pytest.raises(ClientError):
        run(request)
    assert (root / "datasets/test-job/support_aug.npz").exists()
    assert (root / "datasets/test-job/preview_grid.png").exists()
    assert not any(call.kwargs.get("state") == "SUCCESS" for call in states.call_args_list)