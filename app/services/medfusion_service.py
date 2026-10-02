"""
Medfusion 생성 백엔드 — 어댑터.

실제 생성 로직은 bifusion 패키지(`bifusion.generate.medfusion`)에 있다.
이 모듈은 웹 설정(settings)을 패키지의 MedfusionConfig로 매핑하고, 워커가 기대하는
인터페이스(generate_samples / create_preview_grid)를 제공하는 얇은 어댑터일 뿐이다.

- 패키지 import를 위해 settings.BIFUSION_PACKAGE_PATH 를 sys.path에 추가(pip 설치 대신).
- Medfusion 로딩 호환 레시피(streamlit 스텁 / PL 심 / weights_only=False)와 DDIM 루프는
  모두 패키지 쪽(bifusion.generate.medfusion)에 있다.
"""
import math
import sys
from typing import Callable, List, Optional

from PIL import Image

from app.core.config import settings
from app.core.model_loader import get_device


def _ensure_bifusion_on_path() -> None:
    p = settings.BIFUSION_PACKAGE_PATH
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _build_config(image_size: Optional[int] = None):
    """웹 settings → bifusion.generate.medfusion.MedfusionConfig."""
    _ensure_bifusion_on_path()
    from bifusion.generate.medfusion import MedfusionConfig

    latent_shape = tuple(int(x) for x in str(settings.MEDFUSION_LATENT_SHAPE).split(","))
    return MedfusionConfig(
        code_path=settings.MEDFUSION_CODE_PATH,
        weights_dir=settings.MEDFUSION_WEIGHTS_DIR,
        diffusion_ckpt=settings.MEDFUSION_DIFFUSION_CKPT,
        vae_ckpt=settings.MEDFUSION_VAE_CKPT,
        latent_shape=latent_shape,
        guidance_scale=float(settings.MEDFUSION_GUIDANCE_SCALE),
        steps=int(settings.MEDFUSION_STEPS),
        image_size=int(image_size or settings.IMAGE_SIZE),
        channels=int(settings.CHANNELS),
        device=str(get_device()),
    )


class MedfusionService:
    """워커용 어댑터: bifusion 패키지의 MedfusionGenerator를 감싼다."""

    def __init__(
        self,
        model_name: Optional[str] = None,   # 호환용(미사용)
        image_size: Optional[int] = None,
        timesteps: Optional[int] = None,    # 호환용(미사용)
    ):
        _ensure_bifusion_on_path()
        from bifusion.generate.medfusion import MedfusionGenerator

        self._config = _build_config(image_size=image_size)
        self.image_size = self._config.image_size
        self.channels = self._config.channels
        self._generator = MedfusionGenerator(self._config)

    def generate_samples(
        self,
        num_samples: int = 4,
        batch_size: int = 4,
        sampling_steps: Optional[int] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
        condition: Optional[int] = None,
    ) -> List[Image.Image]:
        """합성 이미지 생성 → PIL 리스트 (워커가 기대하는 계약)."""
        return self._generator.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            steps=sampling_steps,
            seed=seed,
            condition=condition,
            progress_callback=progress_callback,
        )

    def augment_with_finetune(
        self,
        images: List[Image.Image],
        num_outputs: int,
        finetune_steps: Optional[int] = None,
        lr: Optional[float] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[Image.Image]:
        """
        업로드 이미지로 UNet을 파인튜닝한 뒤 img2img 증강 → 정확히 num_outputs개 PIL.

        fidelity-first: 업로드 분포에 UNet을 적응시켜 원본에 충실한 변주본을 만든다.
        restore=True로 싱글톤 파이프라인을 원복하므로 클래스/잡 간 오염이 없다.
        """
        if not images:
            raise ValueError("augment_with_finetune: images가 비어 있습니다.")
        v = int(math.ceil(num_outputs / max(1, len(images))))
        out = self._generator.augment_with_finetune(
            images=images,
            finetune_steps=int(finetune_steps if finetune_steps is not None else settings.MEDFUSION_FINETUNE_STEPS),
            lr=float(lr if lr is not None else settings.MEDFUSION_FINETUNE_LR),
            mode="img2img",
            strength=float(strength if strength is not None else settings.MEDFUSION_STRENGTH),
            num_variations_per_image=v,
            seed=seed,
            restore=True,
            progress_callback=progress_callback,
        )
        return out[:num_outputs]

    @staticmethod
    def create_preview_grid(images: List[Image.Image], nrow: int = 4) -> Image.Image:
        """PIL 리스트 → 그리드 이미지 (패키지 구현 위임)."""
        _ensure_bifusion_on_path()
        from bifusion.generate.medfusion import create_preview_grid as _grid
        return _grid(images, nrow=nrow)
