from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from urllib import error, parse, request

from PIL import Image, ImageColor, ImageDraw

from .vision import ensure_rgb

DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiImageError(RuntimeError):
    pass


def _image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    ensure_rgb(image).save(buffer, format="PNG")
    return buffer.getvalue()


def _decode_response_image(encoded: str) -> Image.Image:
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


@dataclass
class ImageGenerationResponse:
    image: Image.Image
    text: str


class NanoBananaClient:
    def mode_label(self) -> str:
        return "mock"

    def generate_preview(self, base_image: Image.Image, prompt: str) -> ImageGenerationResponse:  # pragma: no cover - interface
        raise NotImplementedError

    def edit_crop(self, crop: Image.Image, prompt: str) -> ImageGenerationResponse:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class GeminiNanoBananaClient(NanoBananaClient):
    api_key: str
    model: str = DEFAULT_IMAGE_MODEL
    api_base: str = DEFAULT_API_BASE
    timeout_seconds: int = 90

    @classmethod
    def from_env(cls, model: Optional[str] = None) -> Optional["GeminiNanoBananaClient"]:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        resolved_model = model or os.getenv("AGENT_BANANA_IMAGE_MODEL") or DEFAULT_IMAGE_MODEL
        return cls(api_key=api_key, model=resolved_model)

    def mode_label(self) -> str:
        return self.model

    def generate_preview(self, base_image: Image.Image, prompt: str) -> ImageGenerationResponse:
        preview_prompt = (
            "Generate one draft image preview for localization before region editing. "
            "Keep non-target content stable, do not add extra unrelated changes. "
            f"User edit request: {prompt}"
        )
        return self._generate_with_image(base_image, preview_prompt)

    def edit_crop(self, crop: Image.Image, prompt: str) -> ImageGenerationResponse:
        edit_prompt = (
            "Edit this crop only. Keep the crop edges compatible with the original surroundings. "
            f"Edit request: {prompt}"
        )
        return self._generate_with_image(crop, edit_prompt)

    def _generate_with_image(self, source_image: Image.Image, prompt_text: str) -> ImageGenerationResponse:
        url = f"{self.api_base}/{parse.quote(self.model, safe='')}:generateContent"
        image_bytes = _image_to_png_bytes(source_image)
        parts = [
            {"text": prompt_text},
            {
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            },
        ]
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.4,
                "topP": 0.9,
                "maxOutputTokens": 512,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                raw_payload = response.read().decode("utf-8")
        except error.HTTPError as exc:  # pragma: no cover - depends on live API
            detail = exc.read().decode("utf-8", errors="replace")
            raise GeminiImageError(f"Gemini image request failed with HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:  # pragma: no cover - depends on live API
            raise GeminiImageError(f"Gemini image request failed: {exc.reason}") from exc

        payload = json.loads(raw_payload)
        image, text = self._extract_image(payload)
        if image is None:
            raise GeminiImageError(f"Gemini image response did not include an image: {payload}")
        return ImageGenerationResponse(image=image, text=text)

    def _extract_image(self, payload: dict) -> tuple[Image.Image | None, str]:
        collected_text = []
        for candidate in payload.get("candidates", []):
            content = candidate.get("content") or {}
            for part in content.get("parts", []):
                if part.get("text"):
                    collected_text.append(part["text"].strip())
                inline_data = part.get("inlineData") or part.get("inline_data")
                if inline_data and inline_data.get("data"):
                    return _decode_response_image(inline_data["data"]), "\n".join(chunk for chunk in collected_text if chunk)
        return None, "\n".join(chunk for chunk in collected_text if chunk)


class MockNanoBananaClient(NanoBananaClient):
    def mode_label(self) -> str:
        return "mock-nano-banana"

    def generate_preview(self, base_image: Image.Image, prompt: str) -> ImageGenerationResponse:
        # Keep mock preview visually clean so localization bugs are easier to inspect.
        return ImageGenerationResponse(
            image=ensure_rgb(base_image).copy(),
            text="Mock preview generated for localization.",
        )

    def edit_crop(self, crop: Image.Image, prompt: str) -> ImageGenerationResponse:
        image = ensure_rgb(crop).copy().convert("RGBA")
        width, height = image.size
        color = self._accent(prompt)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rounded_rectangle((4, 4, max(8, width - 4), max(8, height - 4)), radius=16, fill=color + (78,), outline=color + (255,), width=5)
        label = self._label(prompt)
        draw.text((12, max(8, height // 2 - 8)), label, fill=(255, 255, 255, 255))
        edited = Image.alpha_composite(image, overlay).convert("RGB")
        return ImageGenerationResponse(image=edited, text="Mock crop edit applied.")

    def _deterministic_box(self, width: int, height: int, prompt: str) -> tuple[int, int, int, int]:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        box_width = max(48, min(width - 12, width // 3))
        box_height = max(48, min(height - 12, height // 3))
        max_left = max(1, width - box_width - 8)
        max_top = max(1, height - box_height - 8)
        left = 8 + seed % max_left
        top = 8 + (seed // 17) % max_top
        right = min(width - 8, left + box_width)
        bottom = min(height - 8, top + box_height)
        return left, top, right, bottom

    def _accent(self, prompt: str) -> tuple[int, int, int]:
        colors = ["#D97706", "#0F766E", "#BE123C", "#1D4ED8", "#CA8A04"]
        index = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[-2:], 16) % len(colors)
        return ImageColor.getrgb(colors[index])

    def _label(self, prompt: str) -> str:
        lowered = re.sub(r"[^a-z0-9 ]+", " ", prompt.lower())
        words = [word for word in lowered.split() if len(word) > 2]
        if not words:
            return "edit"
        return " ".join(words[:4])[:24]


def build_image_client() -> NanoBananaClient:
    live_client = GeminiNanoBananaClient.from_env()
    if live_client is not None:
        return live_client
    return MockNanoBananaClient()
