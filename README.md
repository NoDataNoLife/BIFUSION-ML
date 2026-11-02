> ⚠️ **최근 대규모 변경(2026-09)**: 생성 백엔드가 흑백 DDPM → **Medfusion**으로 전면 교체되었고, 생성 로직은 **`bifusion` 패키지(`Bifusion_pip-version-Med`)로 이관**되어 웹앱은 `BIFUSION_PACKAGE_PATH`로 이를 import합니다. 또한 실패 시 가짜 결과를 내던 폴백이 모두 **명시적 실패(fail-loud)**로 바뀌었습니다. 배포·프론트 담당은 반드시 [**MIGRATION.md**](./MIGRATION.md)를 먼저 읽으세요.

# BIFUSION-ML

소량 생의학 이미지의 **few-shot 이진 분류 플랫폼**을 위한 AI 모델 서빙 서버 (FastAPI + Celery + Redis).

사용자가 업로드한 소량의 생의학 이미지를 **증강(생성)** → **few-shot 분류 모델 학습** → **신규 이미지 진단**까지 비동기 파이프라인으로 제공한다.

---

## 파이프라인 개요

```
[증강] 원본 소량 이미지 → Medfusion 합성 → support_aug.npz (+ dataset_id)
   │
   ▼
[학습] dataset_id → ProtoNet few-shot 학습 → best.pt (+ model_id, metrics)
   │
   ▼
[진단] model_id + 신규 이미지 → NORMAL / ANOMALY 예측 (+ confidence)
```

- **증강(Augmentation)**: 사전학습 **Medfusion(의료 latent diffusion)**으로 클래스별 합성 이미지를 만들어 원본과 결합, `.npz` 데이터셋 생성.
- **학습(Training)**: 증강 데이터셋으로 **ProtoNet(Prototypical Network)** few-shot 이진 분류기 학습.
- **진단(Inference)**: 학습된 모델로 신규 이미지를 진단하거나, Medfusion 단독 생성 모드로 이미지만 생성.

각 단계는 별도 Celery 큐(`augmentation` / `training` / `inference`)에서 비동기로 실행되고, `job_id`로 상태를 폴링한다.

---

## 구조

```
BIFUSION-ML/
├── app/
│   ├── main.py                       # FastAPI 진입점 (/api/v1/*, /health)
│   ├── routers/                      # 역할별 API (POST /run, GET /{job_id})
│   │   ├── augmentation.py
│   │   ├── training.py
│   │   └── inference.py
│   ├── workers/                      # Celery Worker (큐 분리)
│   │   ├── augmentation_worker.py    # -Q augmentation  (Medfusion 증강)
│   │   ├── training_worker.py        # -Q training      (ProtoNet 학습)
│   │   └── inference_worker.py       # -Q inference     (진단 / 단독 생성)
│   ├── services/
│   │   ├── medfusion_service.py      # 생성 어댑터 (bifusion 패키지에 위임, settings 매핑)
│   │   ├── protonet_service.py       # ProtoNet 학습(train_few_shot) / 진단(predict)
│   │   ├── storage_service.py        # local | s3 산출물 저장
│   │   └── image_utils.py            # base64 ↔ PIL 등
│   ├── schemas/                      # Pydantic 요청/응답 모델
│   └── core/
│       ├── config.py                 # 환경변수 (Medfusion·CHANNELS 등)
│       ├── celery_app.py             # Celery 설정
│       └── model_loader.py           # get_device() + local|s3 체크포인트 경로 해석
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── requirements.txt
└── MIGRATION.md                      # ⚠️ 최근 변경 인수인계 (필독)
```

---

## 생성 백엔드: Medfusion (배포 전 필독)

이 서버의 **유일한** 생성기는 `medical_diffusion`(Medfusion) 기반 latent diffusion이다. 기존 흑백 DDPM은 제거되었고 **config 토글로 되돌릴 수 없다.** 생성 구현은 웹 리포가 아니라 **`bifusion` 패키지(`bifusion.generate.medfusion`)** 에 있으며, `app/services/medfusion_service.py`는 설정을 패키지로 넘기는 얇은 어댑터다. 실행하려면 아래 4가지가 모두 준비돼야 한다.

1. **`bifusion` 패키지 소스** — `Bifusion_pip-version-Med` 폴더. pip 설치 대신 env `BIFUSION_PACKAGE_PATH`를 `sys.path`에 추가해 import한다(`pip install -e .[gpu,medfusion]`로 설치해도 됨). 없으면 워커가 `import bifusion`에서 실패한다.
2. **Python 패키지** — `requirements.txt`에 포함 (`pytorch-lightning, monai, einops, torchio, lpips` 등).
3. **`medical_diffusion` 소스 코드** — pip 패키지가 아니라 디렉터리를 `sys.path`에 추가해 import. 위치는 env `MEDFUSION_CODE_PATH`.
4. **가중치 체크포인트 2개** — env `MEDFUSION_WEIGHTS_DIR` 기준:
   - diffusion: `MEDFUSION_DIFFUSION_CKPT` (기본 `runs/chest_diffusion/last.ckpt`)
   - VAE: `MEDFUSION_VAE_CKPT` (기본 `runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt`)

> ⚠️ `.env.example`의 `MEDFUSION_CODE_PATH` / `MEDFUSION_WEIGHTS_DIR` 기본값은 **개발자 로컬 Windows 절대경로**다. 배포 시 컨테이너 내부 경로로 반드시 교체하고, 해당 코드/가중치를 이미지에 마운트하거나 `COPY` 해야 한다. `docker-compose.yml`의 볼륨에도 추가가 필요하다. 자세한 배포 체크리스트는 [MIGRATION.md §4](./MIGRATION.md) 참고.

> 참고: 현재 생성은 **사전학습 고정 모델 기반의 무조건부(unconditional)** 생성이다. 업로드 배치를 닮은 조건부/파인튜닝 생성은 미구현(향후 과제).

---

## 동작 계약: fail-loud (프론트 필독)

학습된 모델·유효한 데이터·유효한 입력이 **없으면 결과를 지어내지 않고 잡을 실패시킨다.**

| 상황 | 결과 |
|---|---|
| 학습: 데이터셋(`dataset_id`) 없음 | `FileNotFoundError` → 잡 `FAILURE` |
| 진단: 학습 모델(`model_id`) 없음 | `FileNotFoundError` → 잡 `FAILURE` |
| 진단: 유효한 입력 이미지 0장 | `ValueError` → 잡 `FAILURE` |

실패는 `GET /api/v1/.../{job_id}` 응답의 `status="FAILURE"`, `error="<사유 문자열>"`로 노출된다. **프론트는 `FAILURE` 분기에서 `error`를 사용자에게 보여줘야 한다.** (진단 전 학습 성공으로 `model_id` 확보가 선행되어야 함.)

---

## 로컬 실행

```bash
cp .env.example .env
# .env에서 MEDFUSION_CODE_PATH / MEDFUSION_WEIGHTS_DIR 등 실제 경로로 수정
docker compose up
```

- Worker는 큐별로 `--concurrency=1` 권장(특히 GPU 1장에 생성 잡이 동시에 뜨면 불안정).
- `CHANNELS`(1=흑백 / 3=RGB)는 증강·학습·진단 전 구간에서 **일관**되어야 한다.

---

## API

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/v1/augmentation/run` | 증강 job 시작 (`num_samples_per_class`, `normal/anomaly_images_base64`) → `dataset_id` |
| GET  | `/api/v1/augmentation/{job_id}` | job 상태 조회 |
| POST | `/api/v1/training/run` | 학습 job 시작 (`dataset_id`, `k_shot`, `epochs`) → `model_id`, `metrics` |
| GET  | `/api/v1/training/{job_id}` | job 상태 조회 |
| POST | `/api/v1/inference/run` | 진단(`model_id`+`images_base64`) 또는 단독 생성(`num_samples`) |
| GET  | `/api/v1/inference/{job_id}` | job 상태 조회 |
| GET  | `/health` | 헬스체크 |

**공통 응답 `JobResponse`**: `{ job_id, status(PENDING|STARTED|RUNNING|SUCCESS|FAILURE), progress(0-100), result(dict|null), error(str|null) }`

**진단 모드 결정**(`/inference/run`): `images_base64`가 있거나 `num_samples`가 없으면 **진단(ProtoNet)**, `num_samples`가 지정되면 **Medfusion 단독 생성**.

### Swagger
http://localhost:8000/docs

---

## 주요 설정 (`.env` / `app/core/config.py`)

| 변수 | 기본값 | 의미 |
|---|---|---|
| `DEVICE` | `auto` | `auto\|cuda\|cpu\|mps` |
| `IMAGE_SIZE` | `128` | 생성/전처리 해상도 |
| `CHANNELS` | `1` | 1=흑백, 3=RGB (데이터셋·진단·생성 채널 일괄 결정) |
| `BIFUSION_PACKAGE_PATH` | (로컬 경로) | 생성 로직이 담긴 `bifusion` 패키지 폴더 |
| `MEDFUSION_CODE_PATH` | (로컬 경로) | `medical_diffusion` 소스 디렉터리 |
| `MEDFUSION_WEIGHTS_DIR` | (로컬 경로) | 체크포인트 기준 디렉터리 |
| `MEDFUSION_DIFFUSION_CKPT` / `MEDFUSION_VAE_CKPT` | (아래) | diffusion / VAE 가중치 상대경로 |
| `MEDFUSION_LATENT_SHAPE` | `8,32,32` | latent (C,H,W) |
| `MEDFUSION_GUIDANCE_SCALE` | `4.0` | CFG 강도 |
| `MEDFUSION_STEPS` | `50` | 기본 샘플링 스텝 |
| `MODEL_LOADER` / `MODEL_DIR` | `local` / `models` | 체크포인트 로더/경로 |
| `STORAGE_BACKEND` / `LOCAL_STORAGE_DIR` | `local` / `storage` | 산출물 저장 |
| `REDIS_URL` | `redis://localhost:6379/0` | Celery broker |

전체 변경 내역·데이터 포맷·배포 체크리스트는 [**MIGRATION.md**](./MIGRATION.md)를 참고하세요.
