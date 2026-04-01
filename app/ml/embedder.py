from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetModel

from app.logging import get_logger

LOGGER = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoImageProcessor, ResNetModel]:
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetModel.from_pretrained("microsoft/resnet-50")
    model.eval()
    return processor, model


class FaceEmbedder:
    def embed_bgr_face(self, face_bgr: np.ndarray) -> np.ndarray:
        processor, model = _load_model()
        rgb = face_bgr[:, :, ::-1]
        pil_image = Image.fromarray(rgb)
        inputs = processor(pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        vector = outputs.pooler_output.flatten().numpy().astype(np.float32)
        LOGGER.debug("embedding_dim=%s", vector.shape[0])
        return vector
