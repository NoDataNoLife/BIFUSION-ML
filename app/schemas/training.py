from pydantic import BaseModel


class TrainingRequest(BaseModel):
    model_name: str
    dataset_path: str
    epochs: int = 10
    options: dict = {}
