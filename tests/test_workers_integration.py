import pytest
from app.workers.augmentation_worker import run_augmentation
from app.workers.training_worker import run_training
from app.workers.inference_worker import run_inference
from app.services.image_utils import create_synthetic_noise_image, pil_to_base64


def test_end_to_end_3step_pipeline():
    """
    JOB-001 (증강) -> JOB-002 (학습) -> JOB-003 (추론)
    전체 3단계 파이프라인 엔드투엔드 테스트
    """
    # -------------------------------------------------------------
    # 1단계: JOB-001 증강 워커 실행
    # -------------------------------------------------------------
    dummy_imgs = [pil_to_base64(create_synthetic_noise_image(128)) for _ in range(5)]
    aug_req = {
        "num_samples_per_class": 2,  # 빠른 테스트를 위해 2장씩
        "sampling_steps": 5,        # 5타임스텝만
        "normal_images_base64": dummy_imgs,
        "anomaly_images_base64": dummy_imgs,
        "model_name": "best_model.pt",
    }
    aug_res = run_augmentation(aug_req)

    assert aug_res["status"] == "SUCCESS"
    dataset_id = aug_res["dataset_id"]
    assert "preview_url" in aug_res

    # -------------------------------------------------------------
    # 2단계: JOB-002 학습 워커 실행 (1단계 데이터셋 사용)
    # -------------------------------------------------------------
    train_req = {
        "dataset_id": dataset_id,
        "k_shot": 3,
        "epochs": 2,
    }
    train_res = run_training(train_req)

    assert train_res["status"] == "SUCCESS"
    model_id = train_res["model_id"]
    assert "metrics" in train_res
    assert "accuracy" in train_res["metrics"]

    # -------------------------------------------------------------
    # 3단계: JOB-003 추론 워커 실행 (2단계 모델 사용)
    # -------------------------------------------------------------
    test_patient_imgs = [pil_to_base64(create_synthetic_noise_image(128)) for _ in range(2)]
    infer_req = {
        "model_id": model_id,
        "images_base64": test_patient_imgs,
    }
    infer_res = run_inference(infer_req)

    assert infer_res["status"] == "SUCCESS"
    assert infer_res["mode"] == "diagnosis"
    assert len(infer_res["predictions"]) == 2
    assert infer_res["predictions"][0]["predicted_class"] in ("NORMAL", "ANOMALY")
