from typing import List, Optional
from pydantic import BaseModel, Field


class AugmentationRequest(BaseModel):
    num_samples_per_class: int = Field(default=10, description="클래스당 생성할 이미지 수")
    strength: float = Field(default=0.3, ge=0.0, le=1.0, description="원본 충실도↔다양성 (img2img 노이즈 비율; 낮을수록 원본 밀착)")
    finetune_steps: Optional[int] = Field(default=None, description="업로드별 UNet 파인튜닝 스텝 (None=서버 기본). 클수록 적응↑·시간↑")
    steps: Optional[int] = Field(default=None, description="img2img 디노이즈 스텝 (None=서버 기본). 품질↔시간")
    seed: Optional[int] = Field(default=None, description="시드 고정 (재현성)")
    # Base64 인코딩된 이미지 문자열 리스트 (프론트/백엔드 연동용)
    normal_images_base64: Optional[List[str]] = Field(default=[], description="Normal 이미지 Base64 목록")
    anomaly_images_base64: Optional[List[str]] = Field(default=[], description="Anomaly 이미지 Base64 목록")
