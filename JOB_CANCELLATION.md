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
