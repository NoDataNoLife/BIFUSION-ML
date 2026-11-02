# 작업 취소·조회 API

증강·학습·추론에 공통 적용합니다. 이번 취소는 대기 중 작업만 지원합니다.
실행 중 GPU 계산 중단, 재시도 API, 리워드 정산은 포함하지 않습니다.

## BE 연동

`{kind}`는 `augmentation`, `training`, `inference` 중 하나입니다.

| 요청 | 주소 | 동작 |
| --- | --- | --- |
| GET | `/api/v1/{kind}/{job_id}` | 작업 상태·결과 조회 |
| POST | `/api/v1/{kind}/{job_id}/cancel` | 대기 작업 취소. 요청 본문 없음 |

| 상황 | HTTP | 응답 |
| --- | --- | --- |
| 대기 중 취소 / 이미 취소됨 | 200 | `status: CANCELLED` |
| 실행 중 / 성공 / 실패한 작업 취소 | 409 | `detail.error_code: JOB_NOT_CANCELLABLE` |
| 없는 작업 또는 다른 종류의 작업 | 404 | `detail.error_code: JOB_NOT_FOUND` |
| Redis 연결 장애 | 503 | `detail.error_code: JOB_SERVICE_UNAVAILABLE` |
| 성공 기록은 있지만 결과 보관 기간 만료 | 410 | `detail.error_code: JOB_RESULT_EXPIRED` |

GET과 취소의 없는 작업 응답:

```json
{"detail": {"error_code": "JOB_NOT_FOUND", "message": "요청한 작업을 찾을 수 없습니다."}}
```

존재하지 않는 API 주소는 FastAPI 기본 `{"detail":"Not Found"}`를 유지합니다.
BE는 HTTP 404만 보지 말고 `detail.error_code`까지 확인합니다.

## 취소 처리 방식

Redis에 작업 종류와 상태를 등록한 뒤 실행 큐에 넣습니다. Redis는 API와 작업자가 함께 사용하는 작업 기록 저장소입니다.
작업 시작과 취소가 동시에 요청되어도 상태 변경은 한쪽만 성공합니다.
취소가 먼저면 작업자는 계산을 건너뛰고, 시작이 먼저면 취소 요청에 409를 반환합니다.
취소된 메시지는 큐에서 즉시 삭제하지 않으며 작업자가 받았을 때 건너뜁니다.

## 배포 전 확인

- API와 모든 Celery 작업자를 함께 새 버전으로 교체해야 합니다.
- 기존 대기·실행 작업을 먼저 완료하고 새 요청을 잠시 막은 상태에서 교체합니다. 이전 버전에서 접수한 작업은 새 등록 기록이 없어 조회 시 404가 됩니다.
- API와 작업자는 동일한 `REDIS_URL`을 사용해야 합니다. Redis 데이터가 사라지면 작업 존재 여부도 확인할 수 없으므로 서버의 데이터 보존 설정을 확인합니다.
- 등록 기록은 자동 만료하지 않습니다. 결과 본문은 기존 Celery 보관 정책을 따릅니다. 장기 운영 시 기록 정리 정책은 별도로 정해야 합니다.
- 작업자 강제 종료 뒤 자동 복구·재시도는 이번 범위에 없습니다. RUNNING 상태가 남을 수 있습니다.
- RunPod 실서버 검증 전입니다. 배포 후 대기 취소, 실행 중 409, 없는 ID의 404를 다시 확인합니다.

## 로컬 검증

```bash
pip install -r requirements-test.txt
python -m pytest tests/test_job_lifecycle.py tests/test_api_endpoints.py tests/test_augmentation_s3_results.py -q
```

Redis는 테스트용 대체 구현으로 검증합니다. 실제 GPU·AWS·실서버 Redis 검증은 별도입니다.

## 2026-10-17 BE·FE 연결 준비

BE `origin/develop`의 `2f8c147` 소스 기준입니다. 운영 배포와 환경변수 적용 여부는 확인하지 않았습니다.

| FE → BE 요청 | 본문 | 주요 처리 |
| --- | --- | --- |
| POST `/api/v1/jobs/{jobId}/cancel` | 없음 | 작업 생성자 또는 프로젝트 관리자. 409이면 GET으로 상태 재조회 |
| POST `/api/v1/jobs/{jobId}/retry` | 없음 | `retryable=true`일 때 같은 설정·입력으로 새 작업 생성 |
| GET `/api/v1/jobs/{jobId}` | 없음 | BE에 저장된 상태·결과 조회 |

FE는 BE의 숫자 jobId를 사용합니다. BE가 ML의 문자열 job_id로 변환하여 호출합니다. 재시도는 선택 헤더 `Idempotency-Key`(중복 실행 방지 키)를 지원하며, 새 재시도에는 새 키를 쓰고 같은 요청의 네트워크 재전송에는 같은 키를 사용합니다. 성공 응답은 BE의 `CommonResponse<JobResponse>`입니다.

- BE 취소 활성화 설정 `ML_CANCEL_ENABLED`의 기본값은 false이며, 비활성화 상태에서는 새 취소 요청에 503 `J016`을 반환합니다. 이미 취소된 작업은 현재 상태를 반환합니다.
- BE는 ML의 404와 `detail.error_code=JOB_NOT_FOUND`를 함께 검사합니다. 기본 경로 404와 구분하는 점은 현재 ML 구현과 맞습니다.
- 주의: ML은 RUNNING도 409로 거절하지만 BE는 ML 취소의 모든 409를 `J014`(이미 끝났다는 문구)로 변환합니다. 실제로는 실행 중일 수도 있으므로 BE 문구를 “이미 시작되었거나 종료되어 취소할 수 없습니다”로 맞추는 협의가 필요합니다. FE는 409 후 상태를 재조회해야 합니다.
- ML의 410 `JOB_RESULT_EXPIRED`는 BE에서 별도 분류하지 않고 일반 ML 응답 오류로 처리합니다. 결과 만료 시 BE 작업 상태를 어떻게 확정할지 후속 합의가 필요합니다.
- 최신 FE `origin/main`의 `df7aa3e`에서 취소·재시도 경로 및 `retryable` 사용은 검색되지 않았습니다. FE 설정·상태 정리 PR #61의 병합은 확인했지만, 배포 사이트 반영은 확인하지 않았습니다.

배포 순서: 기존 ML 작업 종료 및 새 접수 차단 → API·모든 작업자 동시 교체 → ML 단독 취소·조회 검증 → BE 취소 활성화 및 연동 시험. 실제 서버를 확인하기 전에는 이 순서를 실행하거나 자동 배포가 준비됐다고 간주하지 않습니다.