from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
    dataset_id: str = Field(description="1단계에서 생성된 증강 데이터셋 ID 또는 파일 경로")
    k_shot: int = Field(default=5, description="K-Shot 크기 (1, 3, 5, 10)")
    epochs: int = Field(default=30, description="학습 에포크 수")
    q_query: int = Field(default=5, description="에피소드당 쿼리 수 (고급)")
    episodes_per_epoch: int = Field(default=10, description="에포크당 에피소드 수 (고급)")
    seed: int = Field(default=42, description="재현성 시드 (고급)")
