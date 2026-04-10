from __future__ import annotations

from PIL import Image

from .nano_banana_old import (
    DEFAULT_API_BASE,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    GeminiImageError,
    ImageGenerationResponse,
    NanoBananaClient as _BaseNanoBananaClient,
    GeminiNanoBananaClient as _OldGeminiNanoBananaClient,
    MockNanoBananaClient as _OldMockNanoBananaClient,
    build_image_client as _old_build_image_client,
)
from .vision_old import ensure_rgb


class NanoBananaClient(_BaseNanoBananaClient):
    def edit_full_image(self, image: Image.Image, prompt: str) -> ImageGenerationResponse:  # pragma: no cover - interface
        raise NotImplementedError


class GeminiNanoBananaClient(_OldGeminiNanoBananaClient, NanoBananaClient):
    def edit_full_image(self, image: Image.Image, prompt: str) -> ImageGenerationResponse:
        full_prompt = (
            "Edit the full image according to the instruction. "
            "Keep the scene composition stable and preserve non-target regions. "
            f"Edit request: {prompt}"
        )
        return self._generate_with_image(image, full_prompt)


class MockNanoBananaClient(_OldMockNanoBananaClient, NanoBananaClient):
    def edit_full_image(self, image: Image.Image, prompt: str) -> ImageGenerationResponse:
        base = ensure_rgb(image).copy().convert("RGBA")
        width, height = base.size
        left, top, right, bottom = self._deterministic_box(width, height, prompt)
        color = self._accent(prompt)

        from PIL import ImageDraw

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=color + (64,), outline=color + (220,), width=4)
        proposed = Image.alpha_composite(base, overlay).convert("RGB")
        return ImageGenerationResponse(image=proposed, text="Mock full-image proposal generated.")


def build_image_client() -> NanoBananaClient:
    client = _old_build_image_client()
    if isinstance(client, _OldGeminiNanoBananaClient):
        return GeminiNanoBananaClient(api_key=client.api_key, model=client.model, api_base=client.api_base, timeout_seconds=client.timeout_seconds)
    return MockNanoBananaClient()


__all__ = [
    "DEFAULT_API_BASE",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_TEXT_MODEL",
    "GeminiImageError",
    "ImageGenerationResponse",
    "NanoBananaClient",
    "GeminiNanoBananaClient",
    "MockNanoBananaClient",
    "build_image_client",
]
