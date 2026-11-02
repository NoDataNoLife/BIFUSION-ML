# BIFUSION ML 서비스 — 배포·연동 명세서

> 대상: **백엔드(Spring) 팀** + **서버 세팅 담당**.
> 범위: RunPod(단일 RTX 5090) Pod 세팅 · AI 모델 세팅 · 웹 연동(백엔드 API 명세).
> 작성일: 2026-10-10. 기준: `BIFUSION-ML`(브랜치 `Medfusion`) + `bifusion` 패키지(`Bifusion_pip`, version-Med).

---

## 0. 전체 구조 (먼저 읽기)

```
[사용자 브라우저] → [프론트(React)] → [Spring 백엔드] → [RunPod RTX 5090: ML 서비스]
```

- **ML 서비스는 RunPod GPU 1대(RTX 5090)에서 전부** 돕니다: FastAPI(API) + Celery 워커 3개(증강·학습·진단) + Redis.
- **백엔드(Spring)는 ML 서비스의 공개 URL로 HTTP 요청**만 보냅니다(서버↔서버).
- 모든 작업은 **비동기 잡**입니다: `POST .../run` 으로 제출 → `job_id` 수신 → `GET .../{job_id}` 로 상태 폴링.
- 파이프라인 순서: **증강 → (dataset_id) → 학습 → (model_id) → 진단**.

---

## 준비물 한눈에 (서버 세팅 전 확인)

| 준비물 | 무엇인가 | git 포함? | 서버 위치 / 지정(env) |
|---|---|---|---|
| **BIFUSION-ML 코드** | FastAPI·Celery 워커(서비스 본체) | ✅ git clone | `/workspace/bifusion-ml` |
| **bifusion 패키지** (version-Med) | 생성·분류 로직 — 서비스가 import | ✅ git clone | `/workspace/bifusion-pip` · `BIFUSION_PACKAGE_PATH` |
| **medical_diffusion 코드 폴더** | Medfusion을 불러오고 실행하는 외부 코드 | ❌ **별도 업로드** | `/workspace/medical_diffusion` · `MEDFUSION_CODE_PATH` |
| **eye diffusion `last.ckpt`** (2.3GB) | 학습된 이미지 생성 모델 | ❌ **별도 업로드(볼륨)** | `…/runs/eye_diffusion/last.ckpt` · `MEDFUSION_DIFFUSION_CKPT` |
| **eye VAE `last_vae.ckpt`** (107MB) | 이미지↔latent 변환·복원 | ❌ **별도 업로드(볼륨)** | `…/runs/2022_12_12_224914_eye_vaegan/last_vae.ckpt` · `MEDFUSION_VAE_CKPT` |
| **Python 의존성** | torch **cu128**(5090)·celery·redis·monai·lpips 등 | — (pip) | `pip install -r requirements.txt` + `pip install -e bifusion-pip` |
| **Redis** | 작업 큐(broker) | — (apt) | `apt-get install redis-server` |

- ⚠️ 가중치는 반드시 **eye** 것이어야 함(chest/patho 아님 — chest면 혈액 RGB가 회색조로 뭉개짐).
- ❌ 표시(의 코드·가중치)는 **git에 없음** → 네트워크 볼륨에 업로드 후 env로 경로 지정. ✅ 표시만 git clone.
- 설치·실행 상세는 아래 **PART B(B3~B6)**.

---

# PART A. 백엔드 연동 명세 (Spring 팀용)

## A1. 기본 규약

- **Base URL**: RunPod이 발급하는 공개 URL. 예: `https://<pod-id>-8000.proxy.runpod.net`
  - ⚠️ Pod을 새로 만들면 URL이 바뀝니다 → 백엔드에 **환경변수로** 두세요(하드코딩 금지).
- **인증 없음**: ML 서비스 자체엔 인증이 없습니다. 백엔드↔ML 내부 통신 전제. 외부 노출 시 공유 시크릿 헤더 등은 별도 협의.
- **콘텐츠 타입**: `application/json`. 이미지는 **Base64 문자열**로 전송(data URI 접두어 있어도 됨).
- **공통 응답 `JobResponse`**:
  ```json
  {
    "job_id": "d9c5...-5935",
    "status": "PENDING | STARTED | RUNNING | SUCCESS | FAILED | CANCELLED",
    "progress": 0,
    "result": null,
    "error": null
  }
  ```
  - `status`가 **`SUCCESS`** 면 `result`에 결과 dict, **`FAILED`** 면 `error`에 사유 문자열.
  - (주의) 내부 Celery는 `FAILURE`지만 API 응답에선 **`FAILED`** 로 내려갑니다. 백엔드는 **`FAILED`** 를 실패로 처리하세요.

## A2. 잡 흐름 (한 사이클)

1. **증강** `POST /api/v1/augmentation/run` → `job_id` → 폴링 → `result.dataset_id` 확보
2. **학습** `POST /api/v1/training/run` (위 `dataset_id` 전달) → 폴링 → `result.model_id` 확보
3. **진단** `POST /api/v1/inference/run` (위 `model_id` + 진단 이미지) → 폴링 → `result.predictions`

> 각 단계는 앞 단계의 산출물(`dataset_id` → `model_id`)이 있어야 다음으로 넘어갑니다. **학습 없이 진단하면 실패**합니다.

## A3. 엔드포인트 상세

### (1) 증강 — `POST /api/v1/augmentation/run`
요청 body:
| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `normal_images_base64` | string[] | [] | 정상 업로드 이미지 (클래스당 **최대 10장**) |
| `anomaly_images_base64` | string[] | [] | 비정상 업로드 이미지 (클래스당 **최대 10장**) |
| `num_samples_per_class` | int | 10 | 클래스당 **생성할** 이미지 수 (최대 100) |
| `strength` | float(0~1) | 0.3 | 원본 충실도↔다양성 (낮을수록 원본 밀착) |
| `finetune_steps` | int\|null | null | 업로드별 파인튜닝 스텝 (null=서버 기본) |
| `steps` | int\|null | null | 생성 디노이즈 스텝 (null=서버 기본) |
| `seed` | int\|null | null | 재현성 시드 |

성공 `result` 주요 키:
```json
{ "status":"SUCCESS", "dataset_id":"<id>", "dataset_url":"<npz 경로/URL>",
  "preview_url":"<미리보기 그리드>", "num_normal":110, "num_anomaly":110, "total_samples":220 }
```
> ⚠️ 유효 업로드가 한 클래스라도 0장이면 **실패(`FAILED`)** 합니다(원본 없이 증강 불가).

### (2) 학습 — `POST /api/v1/training/run`
| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `dataset_id` | string | (필수) | 증강 결과의 `dataset_id` |
| `k_shot` | int | 5 | 1/3/5/10 |
| `epochs` | int | 30 | 학습 에포크 |
| `q_query` | int | 5 | (고급) 에피소드당 쿼리 수 |
| `episodes_per_epoch` | int | 10 | (고급) 에포크당 에피소드 |
| `seed` | int | 42 | (고급) 재현성 |

성공 `result`:
```json
{ "status":"SUCCESS", "model_id":"<id>", "model_url":"<best.pt 경로/URL>",
  "dataset_id":"<id>", "k_shot":5,
  "metrics":{ "accuracy":0.95, "f1_score":0.95, "auroc":0.97, "loss":0.02, "epoch":12 } }
```

### (3) 진단 / 단독 생성 — `POST /api/v1/inference/run`
| 필드 | 타입 | 기본 | 설명 |
|---|---|---|---|
| `model_id` | string | "best.pt" | 학습 결과의 `model_id` |
| `images_base64` | string[] | [] | 진단할 이미지 |
| `num_samples` | int\|null | null | **생성 모드**일 때만 사용(장수) |
| `batch_size`,`sampling_steps`,`seed` | — | — | 생성 모드 옵션 |

- **모드 결정**: `images_base64`가 있거나 `num_samples`가 없으면 → **진단(ProtoNet)**. `num_samples`가 지정되면 → **단독 이미지 생성(Medfusion)**.
- 진단 성공 `result`:
```json
{ "status":"SUCCESS", "mode":"diagnosis", "num_diagnosed":2,
  "predictions":[
    {"sample_index":0,"predicted_class":"NORMAL","confidence":0.98,
     "probabilities":{"normal":0.98,"anomaly":0.02}},
    {"sample_index":1,"predicted_class":"ANOMALY","confidence":0.95,
     "probabilities":{"normal":0.05,"anomaly":0.95}} ] }
```

### (4) 헬스체크 — `GET /health` → `{"status":"ok"}`

## A4. 폴링·타임아웃 권장

- 제출 응답은 즉시(`PENDING`) 옵니다. 그 뒤 **1~2초 간격 폴링**으로 `GET /{job_id}` 조회.
- `RUNNING` 중엔 `progress`(0~100)로 진행률 표시 가능.
- **처리 시간 감각(단일 5090 기준)**:
  - 증강: 요청당 **약 30~75초**(파인튜닝 포함). 첫 요청은 모델 로드로 +~10초.
  - 학습/진단: 수 초.
- **동시 요청**: 단일 GPU라 증강은 **순서대로 처리(큐)** 됩니다. 5명이 동시에 제출하면 마지막 사람은 앞 작업이 끝날 때까지 대기 → **UI에 "대기/진행률"** 를 꼭 보여주세요. (자세한 처리량은 Part B 운영 노트)

## A5. 생성 결과 파일 전달

현재 작업 브랜치의 증강은 NPZ·PNG·meta.json을 로컬에 보존하고, S3 게시를 켜면 NPZ·PNG를 추가 업로드합니다. 다음 학습은 로컬 NPZ를 사용합니다.

- 권장 설정: `STORAGE_BACKEND=local` + `AUGMENTATION_PUBLISH_S3=true`.
- `AWS_S3_BUCKET_NAME`, `AWS_REGION`, `S3_OUTPUT_PREFIX`와 업로드 인증·권한을 별도로 설정합니다.
- BE는 `result.artifacts.download`와 `result.artifacts.preview`의 `bucket`, `key`, `size_bytes`를 저장하고 필요할 때 파일 접근 링크를 발급하는 방향입니다. 기존 URL은 S3 게시 시 임시 링크입니다.
- S3 게시 기본값은 false입니다. 로컬 모드의 내부 경로는 브라우저에서 직접 열 수 없습니다.
- 학습·추론 저장 방식은 이번 변경 대상이 아닙니다.
- 코드 구현과 실제 서버 검증은 별개입니다. 현재 RunPod·AWS 검증 전이며, BE의 artifacts 수신 처리는 추가 연동이 필요합니다.

설정·검증 절차와 BE 대조 결과는 [AUGMENTATION_S3_RESULTS.md](AUGMENTATION_S3_RESULTS.md), 취소·동시 배포 주의사항은 [JOB_CANCELLATION.md](JOB_CANCELLATION.md)를 확인합니다.

---
# PART B. 서버 AI 세팅 (RunPod · RTX 5090)

## B1. ⚠️ 5090 호환성 (제일 먼저)

RTX 5090은 **Blackwell(sm_120)** 이라 **최신 CUDA/PyTorch만** 동작합니다.
- Pod 템플릿은 **CUDA 12.8 이상 + PyTorch cu128(2.7+)** 짜리를 선택하세요. (구형 torch는 5090에서 아예 안 돕니다.)
- 설치 시 torch는 반드시 cu128 인덱스로: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

## B2. Pod 생성

RunPod 콘솔 → Deploy:
1. **GPU**: RTX 5090 ×1 (VRAM 32GB — 우리 작업은 요청당 ~4.5GB라 매우 여유).
2. **Template**: CUDA 12.8 + PyTorch cu128 계열.
3. **Volume**: **Network Volume**(영구) 생성 후 `/workspace`에 마운트 → 코드·가중치를 여기에 두면 Pod을 꺼도 유지됩니다. 용량 **최소 50GB**(eye diffusion 2.3GB + 의존성 + 생성물 여유).
4. **Expose HTTP Ports**: `8000` → 이게 공개 URL이 됩니다.
5. Deploy → Running 되면 **Connect → Web Terminal** 접속.

## B3. 코드·가중치 올리기

```bash
cd /workspace
git clone <BIFUSION-ML 주소> bifusion-ml           # 브랜치 Medfusion
git clone <Bifusion_pip 주소> bifusion-pip         # 브랜치 version-Med
# medical_diffusion 코드(35MB) + eye 가중치(아래 2개)를 /workspace 로 업로드
#   medfusion_weights/runs/eye_diffusion/last.ckpt             (2.3GB)
#   medfusion_weights/runs/2022_12_12_224914_eye_vaegan/last_vae.ckpt (107MB)
```
> 큰 파일(2.3GB)은 네트워크 볼륨에 **한 번만** 올리면 됩니다. (Google Drive/HuggingFace에 올려두고 `wget`/`gdown`으로 받아도 됨)

## B4. 의존성 설치

```bash
cd /workspace/bifusion-ml
pip install -r requirements.txt
pip install -e /workspace/bifusion-pip              # bifusion 패키지
apt-get update && apt-get install -y redis-server   # Redis
# torch가 cu128인지 확인: python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

## B5. 환경변수 `/workspace/bifusion-ml/.env`

```dotenv
DEVICE=auto
IMAGE_SIZE=128
CHANNELS=3

BIFUSION_PACKAGE_PATH=/workspace/bifusion-pip
MEDFUSION_CODE_PATH=/workspace/medical_diffusion
MEDFUSION_WEIGHTS_DIR=/workspace/medfusion_weights
MEDFUSION_DIFFUSION_CKPT=runs/eye_diffusion/last.ckpt
MEDFUSION_VAE_CKPT=runs/2022_12_12_224914_eye_vaegan/last_vae.ckpt
MEDFUSION_LATENT_SHAPE=4,32,32
MEDFUSION_GUIDANCE_SCALE=4.0
MEDFUSION_STEPS=50

# 처리량 튜닝(권장): 파인튜닝 스텝을 낮추면 요청당 시간 단축
MEDFUSION_FINETUNE_STEPS=200
MEDFUSION_FINETUNE_LR=0.0001
MEDFUSION_STRENGTH=0.3

REDIS_URL=redis://localhost:6379/0
STORAGE_BACKEND=local
AUGMENTATION_PUBLISH_S3=false  # S3 설정·권한 준비 후 true — A5 참고
LOCAL_STORAGE_DIR=/workspace/storage
```

## B6. 실행 (프로그램 5개)

도커 대신 직접 띄웁니다(RunPod에선 이게 안정적). `/workspace/bifusion-ml` 에 `start.sh`:
```bash
#!/usr/bin/env bash
cd /workspace/bifusion-ml
redis-server --daemonize yes
export $(grep -v '^#' .env | xargs)       # .env 로드
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
celery -A app.core.celery_app.celery_app worker -Q augmentation --concurrency=1 --loglevel=info &
celery -A app.core.celery_app.celery_app worker -Q training     --concurrency=1 --loglevel=info &
celery -A app.core.celery_app.celery_app worker -Q inference    --concurrency=1 --loglevel=info &
wait
```
```bash
chmod +x start.sh && ./start.sh
```

## B7. 공개 URL 확인 & 테스트

- Pod 화면 → **Connect → HTTP 8000** → 공개 URL. 뒤에 `/docs` 붙이면 **Swagger**가 뜹니다(성공).
- 빠른 점검:
  ```bash
  curl https://<pod-id>-8000.proxy.runpod.net/health      # {"status":"ok"}
  ```
- Swagger에서 증강→학습→진단을 순서대로 눌러 전체 흐름 확인.

## B8. 운영 노트 (중요)

- **동시성**: 각 워커 `--concurrency=1` 유지. 단일 GPU에 2잡 이상 몰면 불안정(cuDNN). 증강 요청은 **큐로 순차 처리** → 5명 동시 시 마지막 ~2.5~6분(파인튜닝 스텝·생성 수에 따라). `MEDFUSION_FINETUNE_STEPS=200` + (코드로 생성 배치 상향) 하면 요청당 ~30초대까지 단축 가능.
- **메모리**: 요청당 VRAM ~4.5GB → 32GB 5090은 매우 여유. 메모리는 병목이 아니며, 속도는 "한 번에 한 작업"이라는 점이 결정.
- **비용/영속성**: Pod "Running" 동안 과금 → 안 쓰면 **Stop**. 단, **일반 디스크는 Stop 시 사라지므로 코드·가중치는 Network Volume**에 둬야 보존.
- **URL 변동**: Pod 재생성 시 공개 URL이 바뀜 → 백엔드 환경변수로 관리.
- **CORS**: 백엔드↔ML은 서버↔서버라 불필요. 프론트가 ML을 직접 호출할 경우에만 CORS 설정 추가 필요(현재 없음).

---

## 부록 — 한눈에 체크리스트

- [ ] 5090 Pod 생성(CUDA 12.8/torch cu128), 포트 8000, 네트워크 볼륨 50GB
- [ ] 코드 2개(clone) + medical_diffusion + eye 가중치 2개 업로드
- [ ] `pip install -r requirements.txt` + `pip install -e bifusion-pip` + redis
- [ ] `.env` 작성(경로·eye ckpt·finetune 200)
- [ ] `./start.sh` → `/health` 200 확인 → Swagger로 증강→학습→진단 통과
- [ ] 백엔드에 공개 URL 전달(환경변수) + **생성물 저장 방식(S3 vs 정적 서빙) 합의**(A5)
- [ ] 프론트: 폴링·진행률·`FAILED` 에러 표시 구현
