# BIFUSION-ML

BIFUSION AI 모델 서빙 서버 (FastAPI + Celery)

## 구조

```
BIFUSION-ML/
├── app/
│   ├── main.py                      # FastAPI 진입점
│   ├── routers/                     # 역할별 API 엔드포인트
│   │   ├── augmentation.py          # POST /run, GET /{job_id}
│   │   ├── training.py
│   │   └── inference.py
│   ├── workers/                     # Celery Worker (큐 분리됨)
│   │   ├── augmentation_worker.py   # -Q augmentation
│   │   ├── training_worker.py       # -Q training
│   │   └── inference_worker.py      # -Q inference
│   ├── schemas/                     # Pydantic 요청/응답 모델
│   ├── core/
│   │   ├── config.py                # 환경변수
│   │   ├── celery_app.py            # Celery 설정
│   │   └── model_loader.py          # local | s3 추상화
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

## 로컬 실행

```bash
cp .env.example .env
docker compose up
```

## API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | /api/v1/augmentation/run | Augmentation job 시작 |
| GET  | /api/v1/augmentation/{job_id} | job 상태 조회 |
| POST | /api/v1/training/run | Training job 시작 |
| GET  | /api/v1/training/{job_id} | job 상태 조회 |
| POST | /api/v1/inference/run | Inference job 시작 |
| GET  | /api/v1/inference/{job_id} | job 상태 조회 |
| GET  | /health | 헬스체크 |

## Swagger

http://localhost:8000/docs
