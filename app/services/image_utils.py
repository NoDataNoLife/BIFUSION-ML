import base64
import io
from typing import List, Optional
from PIL import Image
import numpy as np


def base64_to_pil(b64_str: str) -> Image.Image:
    """Base64 문자열(data URI 포함 가능)을 PIL Image로 변환"""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img_data = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_data)).convert("L")
    return img


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """PIL 이미지를 Base64 문자열로 변환"""
    buf = io.BytesIO()
    img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def create_synthetic_noise_image(size: int = 128) -> Image.Image:
    """테스트용 가상 128x128 흑백 노이즈 이미지 생성"""
    arr = np.random.randint(50, 200, (size, size), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")
