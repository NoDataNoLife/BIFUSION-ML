from typing import List, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    model_id: Optional[str] = Field(default="best.pt", description="2단계에서 학습 완료된 분류 모델 ID/경로")
    # 진단용 이미지 (Base64)
    images_base64: Optional[List[str]] = Field(default=[], description="진단할 환자 X-ray 이미지 Base64 목록")
    # 또는 직접 DDPM으로 N장 생성만 하고 싶은 경우 (사진관 모드)
    num_samples: Optional[int] = Field(default=None, description="DDPM 이미지 생성 모드 시 장수")
    batch_size: int = Field(default=4, description="배치 크기")
    sampling_steps: Optional[int] = Field(default=None, description="디퓨전 타임스텝 수")
    seed: Optional[int] = Field(default=None, description="시드값")
