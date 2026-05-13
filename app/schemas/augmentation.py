from pydantic import BaseModel


class AugmentationRequest(BaseModel):
    model_name: str
    dataset_path: str
    options: dict = {}
