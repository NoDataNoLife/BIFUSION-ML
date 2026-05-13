from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "bifusion",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.workers.augmentation_worker",
        "app.workers.training_worker",
        "app.workers.inference_worker",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Seoul",
    task_routes={
        "app.workers.augmentation_worker.*": {"queue": "augmentation"},
        "app.workers.training_worker.*": {"queue": "training"},
        "app.workers.inference_worker.*": {"queue": "inference"},
    },
)
