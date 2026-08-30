import os
from pathlib import Path
import torch
from diffusers import UNet2DModel
from torch_ema import ExponentialMovingAverage


def create_dummy_checkpoint(output_path: str = "models/best_model.pt", image_size: int = 128):
    """
    NDNL-docker UNet2DModel 스펙과 100% 호환되는 더미 가중치 파일 생성
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Creating dummy UNet2DModel checkpoint at {out_file.resolve()}...")

    model = UNet2DModel(
        sample_size=image_size,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    checkpoint = {
        "epoch": 100,
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.state_dict(),
        "val_loss": 0.005432,
        "train_loss": 0.004321,
    }

    torch.save(checkpoint, str(out_file))
    print(f"✅ Successfully created dummy checkpoint: {out_file} (Size: {out_file.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    create_dummy_checkpoint()
