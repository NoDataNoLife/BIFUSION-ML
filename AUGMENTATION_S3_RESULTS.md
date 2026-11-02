# 증강 결과 S3 전달 (BE 공유용)

상태: 코드·모의 테스트 완료, 실제 AWS/RunPod 검증 전. 기준 브랜치: Medfusion.

## 이번 변경

- NPZ, PNG, meta.json을 LOCAL_STORAGE_DIR에 저장하여 다음 학습에서 사용합니다.
- S3 게시를 켜면 다운로드용 support_aug.npz와 미리보기 preview_grid.png만 S3에도 업로드합니다.
- result.artifacts에 파일별 bucket, key, size_bytes(바이트)를 반환합니다.
- 기존 dataset_id, dataset_url, preview_url 필드는 유지합니다. S3 게시 시 기존 URL은 임시 링크이므로 BE는 artifacts의 고정 위치를 보관하고 필요할 때 링크를 발급합니다.
- PNG는 기존과 동일하게 생성 이미지 최대 8장을 모은 이미지입니다. NPZ에는 원본과 생성 이미지가 함께 들어 있습니다. ZIP·개별 이미지 제공·자동 배포는 포함하지 않습니다.

## 서버 설정

증강 결과만 S3에 추가 게시하려면 다음 조합을 사용합니다.

```dotenv
STORAGE_BACKEND=local
AUGMENTATION_PUBLISH_S3=true
LOCAL_STORAGE_DIR=/workspace/storage
AWS_S3_BUCKET_NAME=실제-버킷명
AWS_REGION=실제-리전
S3_OUTPUT_PREFIX=합의한-경로
```

AWS 인증은 서버 환경에 별도로 설정합니다. 실제 키를 저장소에 넣지 않습니다.
API와 Celery 작업자가 같은 환경설정과 저장 경로를 사용하도록 반영 후 재시작합니다.

- 기본값은 AUGMENTATION_PUBLISH_S3=false입니다. local + false이면 기존 로컬 시험을 유지하며 artifacts는 null입니다.
- 기존 STORAGE_BACKEND=s3 설정도 증강에서는 로컬 보존 + S3 게시로 동작합니다. 단, 학습·추론 저장 방식은 이번에 수정하지 않았으므로 위 local + true 구성을 권장합니다.
- 게시가 켜졌는데 버킷이 없으면 생성 전에 실패합니다. 인증·권한 오류는 실제 S3 호출 시 실패합니다.
- 업로드 오류는 삼키지 않으며 성공 결과를 반환하지 않습니다. 로컬 결과는 유지됩니다. 두 파일 중 하나만 S3에 올라간 채 실패할 수 있으며 자동 삭제·재게시 API는 이번 범위에 없습니다. 같은 작업 ID로 재업로드하면 같은 경로를 사용합니다. 새 작업 요청은 새 ID로 생성됩니다.

## 응답 예시 (숫자는 예시)

```json
{
  "dataset_id": "job-id",
  "artifacts": {
    "download": {
      "bucket": "team-bucket",
      "key": "ml-results/datasets/job-id/support_aug.npz",
      "size_bytes": 123456
    },
    "preview": {
      "bucket": "team-bucket",
      "key": "ml-results/datasets/job-id/preview_grid.png",
      "size_bytes": 12345
    }
  }
}
```

BE는 작업 결과 result 내부의 artifacts를 읽습니다. size_bytes는 업로드하는 로컬 파일의 실제 크기입니다. key에는 S3_OUTPUT_PREFIX가 포함됩니다. 기존 URL 필드를 고정 파일 경로로 저장하지 않습니다.

## 검증

```bash
python -m pytest tests/test_augmentation_s3_results.py -q
```

모델 생성과 S3 클라이언트를 테스트용으로 대체하여, 로컬 NPZ 읽기·업로드 바이트 일치·파일 크기·로컬 단독 모드·버킷 누락·두 번째 파일 업로드 실패를 검사합니다. 실제 모델 품질이나 AWS 접근 성공 검증은 아닙니다.

RunPod 접근 복구 후 이 브랜치의 버전을 시험 서버에 적용하고 다음을 확인한 뒤 Medfusion 병합을 진행합니다.

1. 버킷·리전·경로와 ML 업로드 권한, BE 다운로드 권한 확인.
2. 작은 실제 증강 실행 → artifacts의 두 S3 객체 존재·크기 확인.
3. 같은 dataset_id로 다음 학습이 로컬 NPZ를 읽는지 확인.
4. BE에서 미리보기·다운로드 링크를 발급해 파일 확인.

현재 자동 배포는 추가하지 않았으므로 Git 병합만으로 RunPod에 반영되지는 않습니다.
## 2026-10-17 BE 코드 대조

기준: ML `f43e3bd`, BE `origin/develop`의 `2f8c147`. 소스 확인이며 운영 배포·실제 연동 성공을 뜻하지 않습니다.

- BE `MlAugmentationResult`는 기존 `dataset_url`, `preview_url`과 최상위 `dataset_key`, `preview_key`, `bucket`을 받습니다. 중첩된 `artifacts` 및 `size_bytes` 수신 필드는 아직 없습니다.
- BE `JobPollingServiceImpl`은 변환한 저장 키를 기록합니다. `MlJobPoller`는 S3 HEAD 요청으로 파일 정보를 확인하는 경로가 있습니다. 따라서 artifacts 미지원만으로 기존 URL 방식도 모두 실패한다고 단정하지 않습니다.
- 다음 연동은 BE에서 `artifacts.download/preview`를 읽고 허용된 버킷인지 확인한 뒤 저장하는 것입니다. 기존 URL 처리와의 호환 및 파일 크기 검증은 BE 담당자와 맞춥니다.
- RunPod 접근과 새 S3 설정은 아직 전달받지 못했습니다. 실제 업로드·다운로드 검증은 미실시입니다.

실서버 검증은 두 단계로 구분합니다. 먼저 ML API만 직접 호출해 로컬 파일 보존, S3 객체 존재·크기·다운로드, 같은 dataset_id의 학습 입력을 확인합니다. 그다음 BE 링크 발급과 FE 표시를 검증합니다. FE 연결은 ML 단독 검증의 선행 조건이 아닙니다.