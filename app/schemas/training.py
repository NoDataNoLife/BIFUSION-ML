from typing import List, Optional
from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    dataset_id: str = Field(description="1단계에서 생성된 증강 데이터셋 ID 또는 파일 경로")
    k_shot: int = Field(default=5, description="K-Shot 크기 (1, 3, 5, 10)")
    epochs: int = Field(default=30, description="학습 에포크 수")
    query_images_base64: Optional[List[str]] = Field(default=[], description="Query 이미지 Base64 목록")
    has_query_labels: bool = Field(default=False, description="Query 라벨 포함 여부")
