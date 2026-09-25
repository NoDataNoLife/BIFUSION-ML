# MIGRATION.md — BIFUSION-ML 백엔드 변경 인수인계

> 대상 독자: 이 웹 리포(BIFUSION-ML-main)를 이어받는 **개발자 및 AI 에이전트(Claude 등)**.
> 목적: ML 담당 쪽에서 적용한 변경을 **오해 없이** 파악하고, 배포/프론트/에러처리를 안전하게 이어가도록 하는 것.
> 작성일: 2026-09-22 (개정: 2026-09-25 — 생성 로직을 `bifusion` 패키지로 이관). 기준 브랜치: 이 문서와 같은 커밋의 `app/` 트리.

---

## 0. TL;DR (한 문단)

**HTTP API 계약(엔드포인트·요청/응답 스키마·잡 흐름)은 바뀌지 않았다.** 바뀐 것은 세 가지다.
1. **생성기 전면 교체**: 기존 흑백 DDPM(`diffusers.UNet2DModel`) → **Medfusion(`medical_diffusion` latent diffusion)**. config 토글로 되돌릴 수 없음. 새 런타임 의존성(코드+가중치+pip 패키지)이 배포에 반드시 필요.
2. **생성 로직이 `bifusion` 패키지로 이관됨**: 생성 구현은 이제 웹 리포가 아니라 **`bifusion` 패키지(`bifusion.generate.medfusion`)** 에 있다. 웹앱의 `app/services/medfusion_service.py`는 설정을 패키지로 넘기는 **얇은 어댑터**일 뿐이다. 웹앱은 env `BIFUSION_PACKAGE_PATH`를 `sys.path`에 추가해 패키지를 import한다(pip 설치 대신). → 이 패키지 폴더가 배포에 반드시 존재해야 함.
3. **"실패 시 가짜 산출물" 폴백 전면 제거(fail-loud)**: 세 워커가 실패해도 가짜로 "성공"하던 동작이 사라지고, 이제 **예외를 던져 Celery 잡이 `FAILURE`로 끝난다.** 이 실패는 이미 `GET /{job_id}` 응답의 `error` 필드로 노출된다. **프론트가 `FAILURE`/`error`를 사용자에게 보여줘야 한다.**

배포 담당은 §3·§4를, 프론트 담당은 §2·§7을, 코드 읽는 에이전트는 전체를 보라.

---

## 1. 변경 요약 표

| 영역 | 이전(인수 시점) | 지금 | 조치 필요 대상 |
|---|---|---|---|
| 생성 백엔드 | 흑백 1채널 DDPM(`diffusers`) 자체 구현 | Medfusion(`medical_diffusion`) latent diffusion | 배포 |
| 생성 토글 | (사실상 없음) | 없음 — Medfusion 고정, 되돌리기 불가 | 배포/기획 |
| 학습 실패 시 | 랜덤 노이즈로 학습해 "성공" 위장 | `FileNotFoundError` → 잡 `FAILURE` | 프론트 |
| 진단(추론) 모델 없음 | **랜덤 가중치 모델** 만들어 진단 | `FileNotFoundError` → 잡 `FAILURE` | 프론트 |
| 진단(추론) 입력 없음 | **노이즈 이미지**를 진단 | `ValueError` → 잡 `FAILURE` | 프론트 |
| 증강 실패 시 | 랜덤 npz 조작 | Medfusion 정상 생성(조작 폴백 제거) | 프론트 |
| API 스키마/엔드포인트 | — | **동일(불변)** | 없음 |
| 채널 | 흑백 고정 | `CHANNELS`(1|3) 설정화. 기본 1(흑백) | 기획(컬러 원하면 3) |
| pip 의존성 | `torch, diffusers, torch-ema...` | Medfusion 스택 추가(`pytorch-lightning, monai, einops, torchio, lpips...`) | 배포 |
| env 변수 | `DEFAULT_TIMESTEPS, DEFAULT_MODEL_NAME, ALLOW_UNTRAINED_MODEL, GENERATOR_BACKEND` | 삭제됨. 대신 `MEDFUSION_*`, `CHANNELS` 추가 | 배포 |

---

## 2. 동작 계약 변경 — fail-loud (가장 중요)

### 2.1 무엇이 바뀌었나
인수 시점 코드는 **입력/모델/데이터셋이 없거나 깨져도 절대 실패하지 않고**, 랜덤 가중치·랜덤 npz·노이즈 이미지 등 **가짜 산출물을 만들어 잡을 `SUCCESS`로 반환**했다. 겉보기엔 동작했지만, 진단 결과·학습 성적표·증강 데이터가 **무의미(garbage)** 였다.

지금은 세 워커 모두 그런 상황에서 **예외를 던진다.** Celery는 이를 `FAILURE` 상태로 저장한다.

### 2.2 정확한 실패 지점 (예외 타입·조건)

| 워커/파일 | 조건 | 던지는 예외 |
|---|---|---|
| `app/workers/training_worker.py` | `dataset_id`로 `support_aug.npz`를 못 찾음 | `FileNotFoundError`(탐색 경로 명시) |
| `app/workers/inference_worker.py` (모드1 진단) | `model_id`로 학습된 `best.pt`를 못 찾음 | `FileNotFoundError`(탐색 경로 명시) |
| `app/workers/inference_worker.py` (모드1 진단) | 유효하게 디코딩된 입력 이미지가 0장 | `ValueError`(받은 개수 명시) |
| `app/services/protonet_service.py` `train_few_shot` | 특정 클래스 표본이 0개 | `ValueError(f"class {c} has no samples")` |

> 증강 워커(`augmentation_worker.py`)는 원본이 비면 여전히 `create_synthetic_noise_image`로 채워 진행한다(빈 클래스 방어용, 모델/데이터 조작이 아님). 이것까지 fail-loud로 통일할지는 미결(§9).

### 2.3 이 실패가 API로 어떻게 나오나 (프론트 필독)
`GET /api/.../{job_id}` (라우터: `app/routers/{inference,training,augmentation}.py`)는 Celery 상태를 그대로 매핑한다:

```python
elif state == "FAILURE":
    error = str(task.result)      # 예외 메시지 문자열
return JobResponse(job_id=..., status="FAILURE", progress=..., result=None, error=error)
```

즉 **응답 스키마는 그대로**이고, 실패 시 `status="FAILURE"`, `error`에 사람이 읽을 수 있는 사유 문자열이 담긴다. **프론트는 폴링 중 `status=="FAILURE"`면 `error`를 사용자에게 노출**해야 한다(이전엔 이런 실패가 없었으니 UI가 없을 가능성 높음).

예시 `error` 문자열:
- `Trained ProtoNet model not found for model_id=abc123. Searched: [...]`
- `No valid input images to diagnose (received 3 entries, all failed to decode). Provide valid base64-encoded images.`
- `Dataset not found for dataset_id=xyz. Searched: [...]`

### 2.4 프론트가 해야 할 일 (체크리스트)
- [ ] 잡 폴링에서 `FAILURE` 분기 추가, `error` 표시.
- [ ] 진단 호출 전, **학습이 성공(`model_id` 확보)** 됐는지 보장하는 흐름 유지(증강 → 학습 → 진단 순서). 학습 없이 진단하면 이제 실패한다.
- [ ] 업로드 이미지가 실제로 유효한 base64인지(빈 배열/깨진 데이터) UX 상 방지.

---

## 3. 생성기 교체: DDPM → Medfusion (구현은 `bifusion` 패키지)

### 3.1 개념 / 계층
- 이전: `app/services/diffusion_service.py`가 흑백 1채널 3-레벨 `UNet2DModel(64,128,256)` DDPM을 자체 정의·샘플링.
- 지금(2계층):
  - **`bifusion` 패키지** — `bifusion.generate.medfusion` 에 실제 구현(`MedfusionConfig`, `MedfusionGenerator`, `generate_medfusion_samples`, `create_preview_grid`)이 있다. 사전학습 의료 **latent diffusion**(VAE + latent UNet)을 로드·샘플링. 이 패키지는 `Bifusion_pip-version-Med` 폴더에 있고 버전은 `0.5.0+medfusion`.
  - **웹 어댑터** — `app/services/medfusion_service.py`의 `MedfusionService`는 웹 `settings`를 패키지의 `MedfusionConfig`로 매핑하고 워커 인터페이스를 유지하는 **얇은 래퍼**. 로딩 호환/DDIM 루프 같은 무거운 로직은 전혀 없다.
- 두 워커(`augmentation_worker`, `inference_worker` 모드2)는 **동일한 인터페이스** `generate_samples(...) -> List[PIL.Image]`, `create_preview_grid(...)` 로만 호출하므로 워커 로직은 그대로다.

### 3.2 런타임 의존성 (4가지 모두 필요)
1. **`bifusion` 패키지 소스**: `Bifusion_pip-version-Med` 폴더. pip 설치 대신 env `BIFUSION_PACKAGE_PATH`를 `sys.path`에 추가해 import한다(`medfusion_service.py`의 `_ensure_bifusion_on_path`). 배포 이미지에 이 폴더가 포함돼야 함. (`pip install -e .[gpu,medfusion]`로 설치해도 됨.)
2. **Python 패키지**(웹 `requirements.txt`에 포함): `pytorch-lightning>=2.0, monai>=1.2, einops>=0.6, pytorch_msssim>=1.0, torchmetrics>=1.0, torchio>=0.18, lpips>=0.1.4`
3. **`medical_diffusion` 소스 코드**: pip 패키지가 아니라 **디렉터리를 `sys.path`에 추가**해서 import한다. 위치는 env `MEDFUSION_CODE_PATH`. 패키지의 `bifusion.generate.medfusion._load_pipeline`가 `from medical_diffusion.models.pipelines import DiffusionPipeline`를 한다.
4. **가중치 체크포인트 2개**(env `MEDFUSION_WEIGHTS_DIR` 기준 상대경로):
   - diffusion: `MEDFUSION_DIFFUSION_CKPT`(기본 `runs/chest_diffusion/last.ckpt`)
   - VAE: `MEDFUSION_VAE_CKPT`(기본 `runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt`)

### 3.3 로딩 호환 처리(패키지 `bifusion.generate.medfusion`에 내장 — 참고용)
패키지의 `bifusion.generate.medfusion`이 다음을 자동 처리한다(수정 불필요, 원리 이해용):
- `streamlit` 모듈 **스텁 주입**(Medfusion 내부가 `st.progress`를 import하므로). 실제 streamlit 설치 불필요.
- **PyTorch-Lightning 1.x→2.x 심**(`pytorch_lightning.utilities.cloud_io.load` 부재 보완).
- 신뢰 로컬 ckpt이므로 `torch.load(weights_only=False)` 강제(PyTorch 2.6+ 기본값 우회).
- streamlit에 의존하는 `pipeline.sample()/denoise()`를 피하려고 **DDIM 루프를 자체 재구현**(`pipeline.forward` + `latent_embedder.decode`), progress 콜백 연결.
- 파이프라인은 **워커 프로세스당 싱글톤 캐시**(`_PIPE_CACHE`)로 상주 → 첫 호출만 느림(수십 초).

### 3.4 출력 채널
Medfusion 디코더 출력은 `settings.CHANNELS`(1|3)에 맞춰 흑백/RGB PIL로 변환되고 `settings.IMAGE_SIZE`(기본 128)로 리사이즈된다. `_tensor_to_pil` 참고.

---

## 4. 배포 체크리스트 (지금 그대로면 컨테이너에서 실패함)

현재 `Dockerfile`/`docker-compose.yml`은 인수 시점 기준이라 **Medfusion을 위한 준비가 없다.** 아래를 반드시 반영:

- [ ] **`bifusion` 패키지(`Bifusion_pip-version-Med`) 이미지에 포함**: `BIFUSION_PACKAGE_PATH`를 컨테이너 내부 경로로 지정하고 그 폴더를 `COPY`/마운트하거나, 이미지 빌드 시 `pip install -e /path/Bifusion_pip-version-Med[medfusion]`로 설치. 이게 없으면 워커가 `import bifusion`에서 즉시 실패한다.
- [ ] **`medical_diffusion` 코드 이미지에 포함**: `MEDFUSION_CODE_PATH`가 컨테이너 내부 경로를 가리키게 하고, 그 경로에 코드를 `COPY` 하거나 볼륨 마운트. (pip 설치 아님)
- [ ] **가중치 마운트**: `MEDFUSION_WEIGHTS_DIR`를 컨테이너 경로로 지정하고 ckpt 2개를 마운트/복사. `docker-compose.yml`의 `volumes:`에 현재는 `./models:/app/models`만 있음 → **가중치·코드 볼륨 추가 필요.**
- [ ] **env 경로 교체**: `app/core/config.py`의 `MEDFUSION_CODE_PATH`/`MEDFUSION_WEIGHTS_DIR` 기본값은 **개발자 로컬 Windows 절대경로**(`C:\Users\tjdgk\...`)다. 배포 `.env`에서 리눅스 경로로 덮어써야 함.
- [ ] **torch(및 스택) 설치 환경 확인**: `Dockerfile`은 `python:3.11-slim`(CPU). Medfusion은 CPU로도 돌지만 매우 느리다. GPU 배포면 CUDA 베이스 이미지/런타임 필요.
- [ ] **동시성 유지**: compose의 각 worker `--concurrency=1` 유지 권장. 특히 GPU 1장에 생성 잡 2개가 동시에 뜨면 불안정(cuDNN 실패) 가능. 큐 분리(`augmentation`/`training`/`inference`)는 그대로 두면 안전.
- [ ] **(정리, 선택)** `requirements.txt`의 `diffusers>=0.27.0`, `torch-ema>=0.3`는 이제 **미사용**. 제거해도 됨(남겨도 무해).
- [ ] **(정리, 선택)** `.env` 파일에 삭제된 변수(`DEFAULT_TIMESTEPS` 등)가 남아 있으면 제거. `config.py`가 `extra="ignore"`라 에러는 안 나지만 혼란 방지.

첫 기동 시 가중치를 못 찾으면 `bifusion.generate.medfusion._load_pipeline()`이 `FileNotFoundError: Medfusion diffusion checkpoint not found: ...`로 즉시 실패한다(조용히 넘어가지 않음).

---

## 5. 파일 변경 목록 (정확)

### 삭제됨 (import하면 깨짐)
- `app/services/diffusion_service.py`
- `app/services/generator.py`
- `scripts/create_dummy_checkpoint.py`

### 신규
- `app/services/medfusion_service.py` — **얇은 어댑터**. `settings` → `bifusion.generate.medfusion.MedfusionConfig` 매핑 + 워커 인터페이스(`generate_samples`/`create_preview_grid`) 유지. 실제 생성 구현은 없음(패키지 위임).

### 별도 리포/폴더 (웹 리포 밖)
- `Bifusion_pip-version-Med` — `bifusion` 패키지(버전 `0.5.0+medfusion`). 신규 모듈 `bifusion/generate/medfusion.py`에 Medfusion 로딩·DDIM·`MedfusionGenerator`가 있다. `bifusion.generate` / 최상위 `bifusion.generate_medfusion`로 노출, `augment(mode="medfusion")`도 지원. 웹앱은 `BIFUSION_PACKAGE_PATH`로 이 폴더를 import한다.

### 수정됨
- `app/core/config.py` — DDPM 관련 변수 제거, `CHANNELS`·`MEDFUSION_*`·`BIFUSION_PACKAGE_PATH` 추가.
- `app/core/model_loader.py` — `load_ddpm_pipeline`/`_DDPM_MODEL_CACHE`/`diffusers`·`torch_ema` import 제거. `get_device()` + `ModelLoader/LocalModelLoader/S3ModelLoader/get_model_loader`만 유지.
- `app/workers/augmentation_worker.py` — 생성기를 `MedfusionService`로 교체, 채널(`CHANNELS`) 처리, 랜덤 npz 폴백 제거.
- `app/workers/training_worker.py` — 데이터셋 미발견 시 `raise FileNotFoundError`(랜덤 학습 제거).
- `app/workers/inference_worker.py` — 모드1: 모델 미발견 `raise FileNotFoundError`, 입력 없음 `raise ValueError`(랜덤 모델·노이즈 진단 제거). 모드2: `MedfusionService` 사용. 미사용 import(`torch`, `ProtoEncoderConv4`, `create_synthetic_noise_image`) 정리.
- `app/services/protonet_service.py` — `train_few_shot`를 정식 에피소드 few-shot으로 재작성(§8), 인코더 채널을 `settings.CHANNELS`로, `predict` 전처리를 채널 인식형으로.
- `app/services/image_utils.py` — `base64_to_pil`가 색 보존 위해 `.convert("RGB")` 반환(소비자가 `CHANNELS`로 변환).
- `tests/test_pipeline.py` — 삭제된 DDPM 서비스/더미 체크포인트 관련 테스트 제거.
- `requirements.txt`, `.env.example` — 위 변경 반영.

---

## 6. 설정(config) 레퍼런스

`app/core/config.py`(pydantic-settings, `.env` 로드, `extra="ignore"`).

### 제거된 변수 (더 이상 참조 금지)
`DEFAULT_TIMESTEPS`, `DEFAULT_MODEL_NAME`, `ALLOW_UNTRAINED_MODEL`, `GENERATOR_BACKEND`

### 주요 변수 (현행)
| 변수 | 기본값 | 의미 |
|---|---|---|
| `DEVICE` | `auto` | `auto|cuda|cpu|mps` |
| `IMAGE_SIZE` | `128` | 생성/전처리 해상도 |
| `CHANNELS` | `1` | **1=흑백, 3=RGB.** 데이터셋 npz·진단 전처리·생성 출력 채널을 일괄 결정 |
| `BIFUSION_PACKAGE_PATH` | (로컬 절대경로) | 생성 로직이 담긴 `bifusion` 패키지(`Bifusion_pip-version-Med`) 폴더(→`sys.path`) |
| `MEDFUSION_CODE_PATH` | (로컬 절대경로) | `medical_diffusion` 소스 디렉터리(→`sys.path`) |
| `MEDFUSION_WEIGHTS_DIR` | (로컬 절대경로) | 아래 ckpt 상대경로의 기준 디렉터리 |
| `MEDFUSION_DIFFUSION_CKPT` | `runs/chest_diffusion/last.ckpt` | diffusion 체크포인트 |
| `MEDFUSION_VAE_CKPT` | `runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt` | VAE 체크포인트 |
| `MEDFUSION_LATENT_SHAPE` | `8,32,32` | latent (C,H,W) |
| `MEDFUSION_GUIDANCE_SCALE` | `4.0` | classifier-free guidance 강도 |
| `MEDFUSION_STEPS` | `50` | 기본 샘플링 스텝(요청이 없을 때) |
| `MODEL_LOADER` | `local` | `local|s3` |
| `MODEL_DIR` / `MODEL_CACHE_DIR` | `models` / `/tmp/models` | 체크포인트 경로/캐시 |
| `STORAGE_BACKEND` | `local` | `local|s3` |
| `LOCAL_STORAGE_DIR` | `storage` | 로컬 산출물 루트 |

> `CHANNELS`는 반드시 **생성 가중치·학습·진단 전체에서 일관**돼야 한다. 도중에 바꾸면 이전에 만든 npz/모델과 채널이 안 맞아 진단이 틀어진다.

---

## 7. API 계약 (변경 없음 — 프론트 호출부는 그대로)

엔드포인트/스키마는 인수 시점과 동일. 실패 노출(`status/error`)만 실질적으로 의미가 생겼다.

### 잡 흐름
`증강(augmentation)` → `dataset_id` 산출 → `학습(training)`에 `dataset_id` 전달 → `model_id` 산출 → `진단(inference 모드1)`에 `model_id` + 이미지 전달.

### 요청 스키마(발췌, `app/schemas/`)
- 증강 `AugmentationRequest`: `num_samples_per_class:int=10`, `normal_images_base64:[str]`, `anomaly_images_base64:[str]`, `sampling_steps`, `seed`.
- 학습 `TrainingRequest`: `dataset_id`, `k_shot:int=5`, `epochs`, (옵션) `query_images_base64`.
- 진단/생성 `InferenceRequest`: `model_id:str="best.pt"`, `images_base64:[str]`, `num_samples:int|None`.
  - **모드 결정**: `images_base64`가 있거나 `num_samples is None` → **진단(ProtoNet)**. 그 외(`num_samples` 지정) → **Medfusion 단독 생성**.

### 응답 스키마 `JobResponse`
`{ job_id, status(PENDING|STARTED|RUNNING|SUCCESS|FAILURE), progress(0-100), result(dict|null), error(str|null) }`
- `POST .../run` → `{job_id, status:PENDING, progress:0}` 반환, 즉시 비동기.
- `GET .../{job_id}` → 폴링. `SUCCESS`면 `result`에 워커 반환 dict, `FAILURE`면 `error`에 사유.

### 워커 성공 시 `result` 주요 키
- 증강: `dataset_id`, `dataset_url`(support_aug.npz), `preview_url`, `num_normal/num_anomaly/total_samples`.
- 학습: `model_id`, `model_url`(best.pt), `metrics{accuracy,f1_score,auroc,loss,epoch}`.
- 진단: `mode:"diagnosis"`, `predictions:[{sample_index, predicted_class(NORMAL|ANOMALY), confidence, probabilities{normal,anomaly}}]`.
- 생성(모드2): `mode:"generation"`, `image_urls`, `preview_url`.

> 참고: 일부 요청 필드는 현재 **미사용(vestigial)** 이다(예: 증강의 `model_name`). 스키마 호환 위해 남겨둠. 제거하려면 프론트와 합의 후.

---

## 8. 데이터/체크포인트 포맷 (읽는 쪽을 위한 스펙)

### 증강 산출 npz (`datasets/{id}/support_aug.npz`)
- `x`: `(N, C, 128, 128)` float32, 0~1 정규화. `C = CHANNELS`(1 또는 3).
- `y`: `(N,)` int64. 0=Normal, 1=Anomaly.
- `ids`: `(N,)` object 문자열.
- 동봉 `meta.json`: `spec.channels = CHANNELS`.

### ProtoNet 체크포인트 (`models/{id}/best.pt`, `torch.save` dict)
- `encoder_state_dict`: `ProtoEncoderConv4(in_channels=CHANNELS, hidden=64, out_dim=256)` 가중치.
- `prototypes`: `(2, 256)` 텐서(클래스별 프로토타입).
- `k_shot`: int.
- `metrics`: 위 성적표 dict.
> `predict()`는 이 4키 중 `encoder_state_dict`, `prototypes`를 사용. 포맷은 **인수 시점과 동일**(내부 학습 알고리즘만 정식 에피소드 few-shot으로 개선됨).

---

## 9. 알려진 한계 / 후속 작업 (기획·QA·후임 에이전트용)

1. **현재 생성은 무조건부(unconditional)이며 고정 사전학습 모델**이다. Medfusion `chest` 가중치를 그대로 쓰므로, **사용자가 업로드한 배치를 닮은** 샘플이 아니라 **일반 흉부 X-ray 계열** 이미지를 만든다. "업로드 데이터에 맞춘 조건부 생성 / 업로드마다 UNet만 파인튜닝(VAE 동결)"은 **미구현(정식안)**. 데모/QA 시 이 점을 인지할 것.
2. **`CHANNELS` 기본값 1(흑백)**. 이 프로젝트의 방향은 "생의학 이미지의 색/원본 보존(fidelity-first)"이므로 컬러가 중요하면 `CHANNELS=3`으로 두는 것을 검토(단, 생성 가중치/학습/진단 전 구간 일관 필요).
3. **증강 워커의 빈 입력 방어**: 원본이 0장이면 `create_synthetic_noise_image`로 채워 진행한다(fail-loud 아님). 다른 워커들과 정책을 통일할지 미결.
4. **`/tmp/models` 사용**: 학습은 `/tmp/models/{job_id}/best.pt`에 쓰고 진단이 그 경로를 탐색한다. 리눅스 컨테이너면 OK, 로컬 Windows 개발 시 경로 유의.

---

## 10. 배포 후 스모크 체크 (권장 순서)

1. **패키지 import 무결성**(BIFUSION_PACKAGE_PATH 검증):
   ```bash
   python -c "import sys; sys.path.insert(0, '<BIFUSION_PACKAGE_PATH>'); import bifusion; print(bifusion.__version__); from bifusion.generate.medfusion import MedfusionGenerator; print('bifusion medfusion OK')"
   ```
2. **워커 import 무결성**(deleted 모듈 참조 없음 확인):
   ```bash
   python -c "import app.workers.inference_worker, app.workers.training_worker, app.workers.augmentation_worker; print('workers import OK')"
   ```
3. **Medfusion 로드**(가중치/코드 경로 검증): 워커 프로세스에서 `MedfusionService(image_size=128)` 1회 생성 → 로그에 `[Medfusion] Pipeline ready on ...` 뜨는지.
4. **fail-loud 확인**: 존재하지 않는 `dataset_id`로 학습 잡 실행 → `GET /{job_id}`가 `status="FAILURE"`, `error`에 `Dataset not found ...` 나오는지.
5. **정상 경로**: 증강 → (성공한 `dataset_id`로) 학습 → (성공한 `model_id`로) 진단까지 이어지는지.

---

## 부록 A — .env 예시(배포 시 경로만 실제 값으로)

```dotenv
DEVICE=auto
IMAGE_SIZE=128
CHANNELS=1                 # 1=grayscale, 3=RGB

# 생성 로직이 담긴 bifusion 패키지 경로 (웹앱이 sys.path에 추가해 import)
BIFUSION_PACKAGE_PATH=/opt/Bifusion_pip-version-Med

# Medfusion (유일한 생성 백엔드) — 배포 시 컨테이너 내부 경로로 반드시 교체
MEDFUSION_CODE_PATH=/opt/medical_diffusion
MEDFUSION_WEIGHTS_DIR=/opt/medfusion_weights
MEDFUSION_DIFFUSION_CKPT=runs/chest_diffusion/last.ckpt
MEDFUSION_VAE_CKPT=runs/2022_12_12_133315_chest_vaegan/last_vae.ckpt
MEDFUSION_LATENT_SHAPE=8,32,32
MEDFUSION_GUIDANCE_SCALE=4.0
MEDFUSION_STEPS=50

MODEL_LOADER=local
MODEL_DIR=models
MODEL_CACHE_DIR=/tmp/models
STORAGE_BACKEND=local
LOCAL_STORAGE_DIR=storage

REDIS_URL=redis://redis:6379/0
```

## 부록 B — 한 줄 원칙

> **"학습된 모델·유효한 데이터·유효한 입력이 없으면, 결과를 지어내지 말고 실패로 알려라."**
> 이번 변경의 핵심 철학이며, 후속 코드도 이 원칙을 깨지 말 것.
