# RunPod 반복 시작과 연동 테스트

## 확인된 범위

2026-10-21 사용자 캡처와 보고 기준: RTX PRO 4500 인식, 모델 경로 존재,
API health, 인증/Redis 조회, Celery ready/pong, S3 검사 파일 put 성공.
추가 사용자 보고: 증강 작업 67fca68c-301f-4e1c-af73-15959ecddf46의
S3 아티팩트와 BE 다운로드가 확인됐다. 학습 작업
55c846a6-a8ab-47a1-8526-db3dbc51ea78에서도 artifacts.download와
BE 링크로 받은 best.pt의 4,661,015바이트 일치가 확인됐다.
FE 전체 흐름과 다음 부팅 후 복구는 별도 검증 대상이다.
S3 put 성공은 get 권한이나 BE 다운로드 성공을 보장하지 않는다.

## 한 번만 파일 배치

로컬 `scripts/runpod_ops.py`를 다운로드한다. RunPod의 8888 JupyterLab에서
`/workspace` 폴더를 열어 업로드한다. RunPod 웹터미널에서 확인한다.

```bash
ls -l /workspace/runpod_ops.py
```

이 파일에 비밀키는 없다. `.env`는 공유하거나 Git에 추가하지 않는다.
스크립트는 `/workspace/bifusion-ml/.env`의 기존 설정을 읽는다.
실행 위치는 RunPod 웹터미널이다. 사용자 PC 터미널에서 실행하지 않는다.

## 다음 Pod 시작 때

GPU가 할당된 Pod에서 모델 처리까지 시작:

```bash
/workspace/bifusion-venv/bin/python /workspace/runpod_ops.py start --gpu
```

CPU Pod에서 API 테스트만 준비:

```bash
/workspace/bifusion-venv/bin/python /workspace/runpod_ops.py start
```

스크립트는 로컬 Redis가 없으면 root에서 apt로 설치하고, 새로 실행하는 Redis는
`/workspace/redis`에 AOF를 저장한다. 기존 Redis의 보존 설정을 변경하지 않는다.
API 상태/인증/저장소를 검사한다. GPU 모드는 CUDA와 모델 경로도 확인하며
대기열이 비었을 때 하나의 serial Celery worker를 시작한다.
기존 대기 요청을 확인했고 계속 처리하려면 다음 명령을 사용한다.
대기 요청을 삭제하거나 새 요청으로 재전송하지 않는다.

```bash
/workspace/bifusion-venv/bin/python /workspace/runpod_ops.py start --gpu --resume-queued
```
이미 실행 중인 프로그램은 유지한다. `.env` 변경을 자동 반영하지 않는다.
오류가 나면 STOP 메시지와 아래 로그를 확인한다. 반복 실행으로 오류를 덮지 않는다.

이 스크립트는 Pod 생성, GPU 확보, 마이그레이션, HTTP 포트 공개를 하지 않는다.
같은 볼륨이 `/workspace`에 연결되고, 호환되는 Python/torch 컨테이너가 필요하다.
GPU Pod의 HTTP 8000 포트도 별도 공개해야 한다.
이것은 수동 한 번 실행하는 도구이며 Pod 부팅 자동 실행/프로세스 자동 복구 설정은 아니다.

## 평소 상태 확인

```bash
/workspace/bifusion-venv/bin/python /workspace/runpod_ops.py status
```

예상: Redis PONG, API health 200, authenticated lookup 404 JOB_NOT_FOUND.
Worker process IDs는 프로세스 존재만 뜻하며 모델 실행 성공을 뜻하지 않는다.
작업자가 계산 중이면 solo 방식 특성상 inspect ping 응답이 늦을 수 있다.
CPU start는 이미 켜진 작업자를 중지하지 않는다.
외부 브라우저에서도 출력된 ML_BASE_URL 뒤 `/health`를 열어 확인한다.
`start` 성공만으로 모델 계산이나 S3 다운로드가 검증되지는 않는다.

## 현희님과 실제 테스트

1. ML_BASE_URL과 합의한 ML API_KEY를 BE에 설정한다. 키는 별도로 안전하게 전달한다.
2. BE에서 증강 요청 한 건만 보낸다. BE에 허용된 입력 수/파라미터를 지킨다.
3. BE 작업 ID와 ML 응답의 job_id를 구분해 기록한다.
4. 아래 명령으로 작업자 로그를 본다. Ctrl+C는 로그 보기만 종료한다.

```bash
tail -f /workspace/celery-test.log
```

`received`는 전달 성공, `succeeded`는 worker 처리 성공이다.
BE는 ML의 실제 결과 필드를 파싱해야 하며, BE 작업 완료/화면 표시/다운로드까지 따로 확인한다.
ML 직접 조회는 실제 ML job_id로 실행한다.

```bash
/workspace/bifusion-venv/bin/python /workspace/runpod_ops.py job 실제_ML_job_id
```

학습/추론은 각각 `--kind training`, `--kind inference`를 덧붙인다.
API_KEY는 내부에서 읽어 전송하며 출력하지 않는다. 결과의 다운로드 URL에는
임시 서명이 포함될 수 있으므로 공개된 대화나 문서에 그대로 게시하지 않는다.

## 상태별 다음 행동

| 결과 | 다음 행동 |
| --- | --- |
| BE에서 401 | BE와 ML의 API_KEY가 같은지 확인. GitHub/AWS 키와 다르다. |
| BE 요청 후 worker 로그 없음 | BE 요청 응답, ML API 로그, 대기열, 사용 중인 주소 확인 |
| PENDING 지속 | worker 연결/대기열 확인. 같은 요청을 계속 재전송하지 않는다. |
| RUNNING | 작업 로그 확인. 진행률이 한동안 같아도 실패로 단정하지 않는다. |
| FAILED | 같은 ML ID의 error와 worker Traceback을 함께 확인 |
| ML 조회 404 | BE ID와 ML ID 혼동, 잘못된 종류, 다른 Redis/Pod인지 확인 |
| SUCCESS | BE 완료 반영과 실제 다운로드/파일 열기까지 검사 |
| 다운로드 403 | URL 만료, S3 GetObject 권한, BE가 사용하는 버킷/키 확인 |

API 로그:

```bash
tail -n 60 /workspace/uvicorn-current.log
tail -n 100 /workspace/celery-test.log
```

## 테스트 종료

BE 요청을 멈추고 active/reserved 작업이 없는지 확인한다.
테스트용 단일 worker에서는 계산 중 inspect 응답이 없을 수 있으므로 로그/작업 상태도 확인한다.
정상 완료된 결과를 BE에서 확인한 뒤 비용 절감을 원하면 RunPod Stop을 누른다.
사용 중인 Pod/볼륨을 Delete하지 않는다. Stop 중 API는 접속할 수 없다.
Redis AOF/파일 보존은 백업을 대신하지 않으며, 계산 중 중지한 작업의 자동 재개는 보장하지 않는다.

같은 Pod ID와 포트면 주소 구조가 유지된다. 마이그레이션으로 ID가 바뀌면
출력된 새 ML_BASE_URL을 BE에 반영한다. 스크립트가 BE 설정을 자동 변경하지 않는다.
