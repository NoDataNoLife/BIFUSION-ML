import math
from typing import List, Optional, Callable
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

from app.core.model_loader import load_ddpm_pipeline
from app.core.config import settings


class DDPMService:
    def __init__(
        self,
        model_name: Optional[str] = None,
        image_size: Optional[int] = None,
        timesteps: Optional[int] = None,
    ):
        self.image_size = image_size or settings.IMAGE_SIZE
        self.timesteps = timesteps or settings.DEFAULT_TIMESTEPS
        self.model, self.noise_scheduler, self.ema, self.device = load_ddpm_pipeline(
            model_name=model_name,
            image_size=self.image_size,
            timesteps=self.timesteps,
        )

    def generate_samples(
        self,
        num_samples: int = 4,
        batch_size: int = 4,
        sampling_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[Image.Image]:
        """
        지정된 수량만큼 DDPM 합성 이미지를 생성하고 PIL Image 리스트를 반환합니다.
        
        Args:
            num_samples: 총 생성할 이미지 개수
            batch_size: 1회 Denoising 배치 크기
            sampling_steps: 커스텀 타임스텝 (기본: 모델 기본값)
            seed: 랜덤 시드
            progress_callback: fn(current_overall_step, total_overall_steps, progress_percentage)
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        num_steps = sampling_steps or self.noise_scheduler.config.num_train_timesteps
        num_batches = math.ceil(num_samples / batch_size)
        total_steps = num_batches * num_steps

        generated_images: List[Image.Image] = []
        overall_step_counter = 0

        self.model.eval()

        with self.ema.average_parameters():
            with torch.no_grad():
                for batch_idx in range(num_batches):
                    current_bs = min(batch_size, num_samples - batch_idx * batch_size)
                    sample = torch.randn(
                        (current_bs, 1, self.image_size, self.image_size),
                        device=self.device,
                    )

                    for t in reversed(range(num_steps)):
                        t_tensor = torch.full(
                            (sample.size(0),),
                            t,
                            device=self.device,
                            dtype=torch.long,
                        )
                        noise_pred = self.model(sample, t_tensor).sample
                        sample = self.noise_scheduler.step(noise_pred, t, sample).prev_sample

                        overall_step_counter += 1
                        if progress_callback:
                            pct = int((overall_step_counter / total_steps) * 100)
                            progress_callback(overall_step_counter, total_steps, pct)

                    # [0, 1] 범위로 변환
                    sample = (sample + 1.0) / 2.0
                    sample = torch.clamp(sample, 0.0, 1.0)

                    # Tensor (B, 1, H, W) -> List of PIL Images
                    for i in range(current_bs):
                        img_tensor = sample[i, 0].cpu().numpy()
                        img_uint8 = (img_tensor * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_uint8, mode="L")
                        generated_images.append(pil_img)

                    del sample
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return generated_images

    @staticmethod
    def create_preview_grid(images: List[Image.Image], nrow: int = 4) -> Image.Image:
        """PIL 이미지 리스트를 받아 바둑판 모양 그리드 이미지로 결합"""
        if not images:
            raise ValueError("No images provided for preview grid")

        # 모든 이미지를 동일 텐서 형태로 변환
        tensors = []
        for img in images:
            arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
            tensors.append(torch.from_numpy(arr[None, :, :]))  # (1, H, W)

        batch_tensor = torch.stack(tensors)  # (N, 1, H, W)
        grid_tensor = make_grid(batch_tensor, nrow=nrow, padding=2, normalize=False)
        # grid_tensor: (3, Grid_H, Grid_W) or (1, Grid_H, Grid_W)
        if grid_tensor.shape[0] == 3:
            grid_np = (grid_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            grid_np = (grid_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

        return Image.fromarray(grid_np, mode="L")
