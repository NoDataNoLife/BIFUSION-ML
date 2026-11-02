from celery import Task
from celery.exceptions import Ignore
from app.core import job_registry


class ManagedTask(Task):
    """Skip cancelled/duplicate deliveries before running any model code."""
    abstract = True

    def before_start(self, task_id, args, kwargs):
        if not job_registry.transition(task_id, "PENDING", "RUNNING"):
            raise Ignore()

    def on_success(self, retval, task_id, args, kwargs):
        job_registry.transition(task_id, "RUNNING", "SUCCESS")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_registry.transition(task_id, "RUNNING", "FAILED")
