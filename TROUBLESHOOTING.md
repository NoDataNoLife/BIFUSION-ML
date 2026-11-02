# TROUBLESHOOTING — 모델/서비스 실패 로그 가이드

> 대상: 서버 운영 담당 + 백엔드/프론트. BIFUSION-ML 서비스에서 날 수 있는 실패를 **종류별**로 정리.
> 작성일: 2026-10-13.

---

## 0. 실패가 "어디서" 드러나나 (먼저 읽기)

실패는 3가지 경로로 나타납니다:

| 경로 | 어떻게 보이나 | 어떤 실패가 여기로 |
|---|---|---|
| **(a) 잡 실패** | `GET /api/v1/.../{job_id}` → `status="FAILED"`, `error="<사유>"` | **대부분** (입력/모델/가중치/GPU/import 등 작업 중 예외) |
| **(b) 제출 실패** | `POST .../run` 이 5xx 또는 연결 오류 | 인프라(특히 Redis 끊김), 요청 스키마 오류(422) |
| **(c) 멈춤** | `status`가 `PENDING`에서 안 바뀜(에러도 없음) | 워커 미기동 / 큐 이름 불일치 → **타임아웃 처리 필요** |

> 핵심: **코드 레벨 실패는 거의 모두 (a)** 로 떨어지고 `error` 문자열에 사유가 담깁니다. 백엔드는 `FAILED` + `error`만 잘 노출하면 됩니다. (b)·(c)는 인프라 점검 대상.

로그는 **각 Celery 워커의 stdout**(augmentation/training/inference)과 **uvicorn** 콘솔에서 봅니다.

---

## 1. 입력·비즈니스 실패 — 정상 설계(fail-loud) · `status=FAILED`

사용자 입력/순서 문제. **버그 아님** — 지어내지 않고 명확히 실패시키는 것. 프론트가 `error`를 그대로 보여주면 됩니다.

| 단계 | 에러 메시지(예시) | 원인 | 조치 |
|---|---|---|---|
| 증강 | `No valid Normal images to augment (received 0 entries, all failed to decode).` | 정상 이미지 0장(또는 전부 디코드 실패) | 정상 이미지 업로드/형식 확인 |
| 증강 | `No valid Anomaly images to augment (received N entries, all failed to decode).` | 비정상 이미지 0장 | 비정상 이미지 업로드 확인 |
| 학습 | `Dataset not found for dataset_id=xxx. Searched: [...]` | 잘못된/만료된 `dataset_id`, 증강 미완료 | 증강 성공 후의 `dataset_id` 전달 |
| 학습 | `train_few_shot: class c has no samples in dataset` | 특정 클래스 표본 0 | 증강 데이터가 두 클래스 모두 포함되는지 확인 |
| 진단 | `Trained ProtoNet model not found for model_id=xxx. Searched: [...]` | 잘못된 `model_id`, 학습 미완료 | 학습 성공 후의 `model_id` 전달(학습 선행) |
| 진단 | `No valid input images to diagnose (received N entries, all failed to decode). Provide valid base64-encoded images.` | 진단 이미지 0장/깨짐 | 유효 base64 이미지 전송 |

> 프론트 체크리스트: **증강 → 학습 → 진단 순서 보장**, 빈 업로드/깨진 base64 사전 방지.

---

## 2. 셋업·의존성 실패 — 주로 **첫 요청 시** `FAILED` (import가 지연 로드라)

배포 경로/설치 문제. 워커가 처음 그 작업을 돌릴 때 예외 → `FAILED`.

| 로그(예시) | 원인 | 조치 |
|---|---|---|
| `ModuleNotFoundError: No module named 'bifusion'` | `BIFUSION_PACKAGE_PATH` 잘못 / 패키지 미설치 | 경로 확인, `pip install -e /workspace/bifusion-pip` |
| `ModuleNotFoundError: No module named 'medical_diffusion'` | `MEDFUSION_CODE_PATH` 잘못 | medical_diffusion 폴더 위치/경로 확인 |
| `FileNotFoundError: Medfusion diffusion checkpoint not found: /workspace/.../last.ckpt` | eye diffusion 가중치 없음/경로 오타 | `MEDFUSION_WEIGHTS_DIR`+`MEDFUSION_DIFFUSION_CKPT` 확인, 파일 업로드 |
| `Model checkpoint not found at: ...` (model_loader) | 모델 로더 경로 문제 | `MODEL_DIR`/스토리지 경로 확인 |
| `ValidationError ... for Settings` | `.env` 값 타입/형식 오류 | `.env` 문법·숫자 필드 확인 |

> 빠른 자가진단: `python -c "import sys; sys.path.insert(0,'/workspace/bifusion-pip'); import bifusion; from bifusion.generate.medfusion import MedfusionGenerator; print('ok')"`

---

## 3. GPU·런타임 실패

| 로그(예시) | 원인 | 조치 |
|---|---|---|
| `CUDA error: no kernel image is available for execution on the device` | **5090(Blackwell)에 구형 torch** (sm_120 미지원) | torch를 **cu128**로 재설치: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` |
| `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...` | 배치/동시작업 과다 | 생성 batch 축소, 동시 GPU 작업 금지(concurrency=1), 입력 장수 확인 |
| `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED` | **한 GPU에서 2잡 동시 실행** | 워커 `--concurrency=1` 유지, GPU에 작업 하나씩 |
| `RuntimeError: No CUDA GPUs are available` | Pod에 GPU 미할당 / 드라이버 문제 | GPU Pod 확인, `nvidia-smi` |

---

## 4. 모델·설정 불일치 (가중치/채널/latent)

| 증상/로그 | 원인 | 조치 |
|---|---|---|
| 생성물이 **회색조**로 뭉개짐(에러는 아님) | **chest 가중치**를 RGB 데이터에 사용 | 반드시 **eye** 가중치 사용(`runs/eye_diffusion`, `..._eye_vaegan`) |
| `RuntimeError: ... size mismatch ...` / 채널 불일치 | ckpt와 `MEDFUSION_LATENT_SHAPE` 불일치(eye=`4,32,32`) | eye 조합으로 통일: diffusion=eye, vae=eye, latent=`4,32,32` |
| 진단이 이상/채널 오류 | `CHANNELS` 불일치(증강/학습/진단 간) | 전 구간 `CHANNELS=3` 일관 유지 |
| 파이프라인 로드 중 네트워크 오류(vgg/LPIPS 다운로드) | **오프라인 서버**에서 LPIPS가 vgg 가중치 다운로드 시도 | 서버 인터넷 허용, 또는 vgg 가중치 사전 캐시 |

---

## 5. 인프라 (Redis·스토리지·디스크)

| 증상/로그 | 원인 | 조치 |
|---|---|---|
| `POST /run` 5xx / `redis ... ConnectionError` / 잡이 계속 `PENDING` | **Redis 미기동** 또는 `REDIS_URL` 오류 | `redis-server` 실행 확인, `REDIS_URL` 점검 |
| 잡이 영원히 `PENDING` (에러 없음) | **해당 큐 워커 미기동** / 큐 이름 불일치 | augmentation/training/inference 워커 3개 모두 떠 있는지 확인 |
| `botocore ... NoCredentialsError` / `NoSuchBucket` | `STORAGE_BACKEND=s3`인데 자격증명/버킷 설정 누락 | AWS 키·버킷 설정 확인(또는 local로) |
| `OSError: [Errno 28] No space left on device` | 디스크 가득(생성물/가중치) | 볼륨 용량 확보(50GB 권장), 오래된 산출물 정리 |

---

## 6. 빠른 진단 순서 (위에서 아래로)

1. **`/health` 200인가?** 아니면 서버(uvicorn) 자체 미기동.
2. **잡이 `PENDING`에서 안 변하나?** → Redis·워커 3개 기동 확인(§5).
3. **`FAILED` + `error` 메시지 확인** → §1(입력)인지 §2~4(셋업/GPU/설정)인지 분류.
4. **자가진단 import 한 줄**(§2) 실행 → 패키지/코드/가중치 경로 검증.
5. **`nvidia-smi` + torch cu128 확인**(§3) → 5090 호환.

---

## 부록 — 로그 보는 곳

- **uvicorn 콘솔**: API 요청/422 스키마 오류.
- **각 Celery 워커 stdout**: 실제 작업 예외 traceback(가장 중요). `--loglevel=info`로 실행.
- `[Medfusion] Loading pipeline...` / `[Medfusion] Pipeline ready on cuda` 로그가 **첫 증강/생성 요청 때** 1회 떠야 정상(안 뜨면 로드 실패).
- Redis: `redis-cli ping` → `PONG` 이면 정상.
