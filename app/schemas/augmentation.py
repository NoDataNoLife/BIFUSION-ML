from typing import List, Optional
from pydantic import BaseModel, Field


class AugmentationRequest(BaseModel):
    num_samples_per_class: int = Field(default=10, description="클래스당 생성할 이미지 수")
    sampling_steps: int = Field(default=100, description="디퓨전 타임스텝 수 (100 표준)")
    guidance_scale: float = Field(default=7.5, description="조건 반영 강도 (7.5 권장)")
    seed: Optional[int] = Field(default=None, description="시드 고정 (재현성)")
    model_name: Optional[str] = Field(default="best_model.pt", description="사용할 DDPM 체크포인트")
    # Base64 인코딩된 이미지 문자열 리스트 (프론트/백엔드 연동용)
    normal_images_base64: Optional[List[str]] = Field(default=[], description="Normal 이미지 Base64 목록")
    anomaly_images_base64: Optional[List[str]] = Field(default=[], description="Anomaly 이미지 Base64 목록")
