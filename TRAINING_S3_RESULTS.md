# Training download artifacts

Training retains best.pt under LOCAL_STORAGE_DIR/models/{job_id}/best.pt
for inference. When TRAINING_PUBLISH_S3=true (or STORAGE_BACKEND=s3), it
also uploads a copy and returns result.artifacts.download with bucket,
key and size_bytes. model_url remains the local path. BE creates download
URLs from artifact metadata; ML does not generate a model download URL.

Required deployment settings:

```dotenv
STORAGE_BACKEND=local
TRAINING_PUBLISH_S3=true
S3_OUTPUT_PREFIX=ml-results
```

Keep existing bucket, region, credentials, model paths and API key.
Upload requires S3 PutObject on ml-results/models/*; BE's signing identity
needs GetObject. Do not make the bucket public.

Deploy app/core/config.py and app/workers/training_worker.py together.
After pending/running work has drained, restart the Celery worker to load
code and settings. Do not interrupt an active training task. Use the
existing worker command and its augmentation,training,inference queues.
The API routing/response contract does not change.

Submit a new training job using an existing dataset. Confirm:

- SUCCESS with artifacts.download.key = ml-results/models/{job_id}/best.pt.
- Local best.pt still exists for inference.
- BE can download the S3 copy and its byte count matches size_bytes.

Existing completed job results do not gain artifacts retroactively.
An upload failure fails the job but leaves the local checkpoint in place.
Automated tests mock training and S3; they do not verify a deployed GPU job.
