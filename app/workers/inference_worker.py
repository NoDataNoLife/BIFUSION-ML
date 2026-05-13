from app.core.celery_app import celery_app
import time


@celery_app.task(bind=True, name="app.workers.inference_worker.run_inference")
def run_inference(self, request: dict):
    """
    Inference job worker.
    TODO: 실제 inference 로직으로 교체
    """
    self.update_state(state="STARTED", meta={"progress": 0})

    # Mock: 실제 로직으로 교체
    time.sleep(5)

    return {"status": "done", "request": request}
