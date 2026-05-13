from app.core.celery_app import celery_app
import time


@celery_app.task(bind=True, name="app.workers.augmentation_worker.run_augmentation")
def run_augmentation(self, request: dict):
    """
    Augmentation job worker.
    TODO: 실제 augmentation 로직으로 교체
    """
    self.update_state(state="STARTED", meta={"progress": 0})

    # Mock: 실제 로직으로 교체
    time.sleep(5)
    self.update_state(state="STARTED", meta={"progress": 50})
    time.sleep(5)

    return {"status": "done", "request": request}
