"""
Medfusion(의료 latent diffusion) 생성 백엔드 — 이 프로젝트의 유일한 생성기.

워커가 기대하는 인터페이스(generate_samples / create_preview_grid)를 제공한다
(기존 흑백 DDPM은 제거되고 Medfusion으로 완전히 대체됨).

- 검증된 로딩 레시피 embed: streamlit 스텁 + PL 1.x→2.x 심 + torch.load(weights_only=False)
- streamlit에 의존하는 pipeline.sample()/denoise() 대신, concurrency 안전을 위해
  denoise 루프를 pipeline.forward + latent_embedder.decode로 자체 재구현(progress_callback 연결)
- 출력은 settings.CHANNELS(1|3)에 맞춰 흑백/RGB PIL로 변환, settings.IMAGE_SIZE로 리사이즈
"""
import contextlib
import math
import os
import sys
import types
from typing import Callable, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

from app.core.config import settings
from app.core.model_loader import get_device


# ==========================================
# 호환 심 (medical_diffusion import 전에 1회)
# ==========================================
def _install_compat_shims() -> None:
    # 1) streamlit 스텁 (denoise의 st.progress; __file__은 반드시 문자열)
    if "streamlit" not in sys.modules:
        st = types.ModuleType("streamlit")
        st.__file__ = "<stub-streamlit>"

        class _Bar:
            def progress(self, *a, **k):
                return None

            def empty(self, *a, **k):
                return None

        st.progress = lambda *a, **k: _Bar()
        sys.modules["streamlit"] = st

    # 2) PL 1.x → 2.x 심 (import만 통과시키면 됨: load_from_checkpoint 경로엔 미사용)
    if "pytorch_lightning.utilities.cloud_io" not in sys.modules:
        cio = types.ModuleType("pytorch_lightning.utilities.cloud_io")
        cio.load = lambda p, map_location=None: torch.load(
            p, map_location=map_location, weights_only=False
        )
        sys.modules["pytorch_lightning.utilities.cloud_io"] = cio
        try:
            import pytorch_lightning.utilities as _plu
            _plu.cloud_io = cio
            import pytorch_lightning.utilities.migration as _mig
            if not hasattr(_mig, "pl_legacy_patch"):
                @contextlib.contextmanager
                def _legacy_patch():
                    yield
                _mig.pl_legacy_patch = _legacy_patch
        except Exception:
            pass


@contextlib.contextmanager
def _trusted_torch_load():
    """신뢰 로컬 Medfusion ckpt 로드용 (PyTorch 2.6+ weights_only 기본 True 우회)."""
    orig = torch.load

    def patched(*a, **k):
        k["weights_only"] = False
        return orig(*a, **k)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


# ==========================================
# 파이프라인 싱글톤 (워커 프로세스 메모리 상주)
# ==========================================
_PIPE_CACHE: dict = {}


def _load_pipeline():
    key = (settings.MEDFUSION_DIFFUSION_CKPT, settings.MEDFUSION_VAE_CKPT)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]

    _install_compat_shims()
    if settings.MEDFUSION_CODE_PATH not in sys.path:
        sys.path.insert(0, settings.MEDFUSION_CODE_PATH)

    from medical_diffusion.models.pipelines import DiffusionPipeline

    device = get_device()
    diff_ckpt = os.path.join(settings.MEDFUSION_WEIGHTS_DIR, settings.MEDFUSION_DIFFUSION_CKPT)
    vae_ckpt = os.path.join(settings.MEDFUSION_WEIGHTS_DIR, settings.MEDFUSION_VAE_CKPT)
    if not os.path.exists(diff_ckpt):
        raise FileNotFoundError(f"Medfusion diffusion checkpoint not found: {diff_ckpt}")

    print(f"[INFO] Loading Medfusion pipeline: {diff_ckpt}")
    with _trusted_torch_load():
        pipe = DiffusionPipeline.load_from_checkpoint(
            diff_ckpt, latent_embedder_checkpoint=vae_ckpt
        )
    pipe.to(device).eval()
    _PIPE_CACHE[key] = (pipe, device)
    print(f"[INFO] Medfusion pipeline ready on {device}")
    return pipe, device


class MedfusionService:
    """Medfusion latent diffusion 생성 서비스 (프로젝트의 유일한 생성기)."""

    def __init__(
        self,
        model_name: Optional[str] = None,   # 호환용(미사용; ckpt는 settings로 지정)
        image_size: Optional[int] = None,
        timesteps: Optional[int] = None,    # 호환용(미사용; Medfusion은 자체 스케줄러)
    ):
        self.image_size = image_size or settings.IMAGE_SIZE
        self.channels = int(settings.CHANNELS)
        self.latent_shape = tuple(int(x) for x in str(settings.MEDFUSION_LATENT_SHAPE).split(","))
        self.guidance_scale = float(settings.MEDFUSION_GUIDANCE_SCALE)
        self.pipe, self.device = _load_pipeline()

    @torch.no_grad()
    def generate_samples(
        self,
        num_samples: int = 4,
        batch_size: int = 4,
        sampling_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        condition: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Medfusion으로 합성 이미지 생성 → PIL 리스트 반환 (워커가 기대하는 계약).

        condition=None 이면 무조건부 생성. 정수(예: 0/1)면 해당 클래스 조건부(사전학습 라벨 기준).
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        steps = int(sampling_steps or settings.MEDFUSION_STEPS)
        num_batches = math.ceil(num_samples / batch_size)
        total_steps = num_batches * steps

        pipe = self.pipe
        sched = pipe.noise_scheduler
        results: List[Image.Image] = []
        done = 0

        for b in range(num_batches):
            bs = min(batch_size, num_samples - b * batch_size)
            cond = None
            if condition is not None:
                cond = torch.tensor([int(condition)] * bs, device=self.device)

            x_t = sched.x_final(torch.zeros((bs, *self.latent_shape), device=self.device))
            ts = torch.linspace(0, sched.T - 1, steps, dtype=torch.long, device=self.device)
            self_cond = None

            for i, t in enumerate(reversed(ts)):
                x_t, x_0, x_T, self_cond = pipe(
                    x_t, t.expand(bs), cond,
                    self_cond=self_cond, guidance_scale=self.guidance_scale, un_cond=None,
                )
                self_cond = self_cond if pipe.use_self_conditioning else None

                # DDIM 업데이트 (마지막 스텝 제외)
                if (steps - i - 1) > 0:
                    t_next = ts[steps - i - 2]
                    alpha = sched.alphas_cumprod[t]
                    alpha_next = sched.alphas_cumprod[t_next]
                    sigma = ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()
                    x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * torch.randn_like(x_t)

                done += 1
                if progress_callback:
                    progress_callback(done, total_steps, int((done / total_steps) * 100))

            # latent → 픽셀 [-1,1]
            imgs = pipe.latent_embedder.decode(x_t) if pipe.latent_embedder is not None else x_t
            imgs = ((imgs.clamp(-1, 1) + 1) / 2).cpu()  # [0,1] (bs, C, H, W)
            for i in range(bs):
                results.append(self._tensor_to_pil(imgs[i]))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _tensor_to_pil(self, arr: torch.Tensor) -> Image.Image:
        """(C,H,W) [0,1] → settings.CHANNELS 모드 PIL, image_size로 리사이즈."""
        np_img = (arr.numpy() * 255).clip(0, 255).astype(np.uint8)
        if np_img.shape[0] == 1:
            pil = Image.fromarray(np_img[0], mode="L")
        else:
            pil = Image.fromarray(np.transpose(np_img, (1, 2, 0)), mode="RGB")
        pil = pil.convert("L" if self.channels == 1 else "RGB")
        if pil.size != (self.image_size, self.image_size):
            pil = pil.resize((self.image_size, self.image_size))
        return pil

    @staticmethod
    def create_preview_grid(images: List[Image.Image], nrow: int = 4) -> Image.Image:
        """PIL 리스트 → 그리드 이미지 (흑백/RGB 자동)."""
        if not images:
            raise ValueError("No images provided for preview grid")
        rgb = images[0].mode == "RGB"
        tensors = []
        for img in images:
            im = img.convert("RGB" if rgb else "L")
            arr = np.array(im, dtype=np.float32) / 255.0
            if rgb:
                tensors.append(torch.from_numpy(np.transpose(arr, (2, 0, 1))))  # (3,H,W)
            else:
                tensors.append(torch.from_numpy(arr[None, :, :]))               # (1,H,W)
        grid = make_grid(torch.stack(tensors), nrow=nrow, padding=2, normalize=False)
        if rgb:
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            return Image.fromarray(grid_np, mode="RGB")
        grid_np = (grid[0].cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(grid_np, mode="L")
