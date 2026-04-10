"""VLM Critic — Semantic verification of edit quality.

Sits outside the executor loop and verifies whether the edit instruction
was actually fulfilled by examining the result image with a VLM.

Uses interleaved text+image prompting to ensure the model can clearly
distinguish original from result, with chain-of-thought forced comparison.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib import error, parse, request

from PIL import Image

from .logging_config import log_function

logger = logging.getLogger(__name__)


@dataclass
class CriticVerdict:
    """Result of a VLM Critic evaluation."""
    fulfilled: bool
    confidence: float
    semantic_score: float
    reasoning: str
    residual_objects: List[str]
    suggestions: List[str]

    def to_dict(self) -> dict:
        return {
            "fulfilled": self.fulfilled,
            "confidence": round(self.confidence, 3),
            "semantic_score": round(self.semantic_score, 3),
            "reasoning": self.reasoning,
            "residual_objects": self.residual_objects,
            "suggestions": self.suggestions,
        }


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _call_gemini(api_key: str, model: str, text_prompt: str,
                 images: list, temperature: float = 0.1) -> str:
    """Call Gemini generateContent API with text + images (simple case)."""
    parts = [{"text": text_prompt}]
    for img_b64 in images:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": img_b64,
            }
        })

    body = json.dumps({
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "text/plain",
            "temperature": temperature,
            "maxOutputTokens": 1024,
        },
    }).encode("utf-8")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{parse.quote(model, safe='')}:generateContent"
    )
    req = request.Request(url, data=body, method="POST",
                          headers={
                              "Content-Type": "application/json; charset=utf-8",
                              "x-goog-api-key": api_key,
                          })

    with request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("candidates", [])
    if not candidates:
        return ""
    parts_out = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts_out)


class VLMCritic:
    """Uses Gemini as a VLM to verify edits semantically.

    Key design: images are INTERLEAVED with text labels as separate
    message parts, so the model can clearly distinguish original from result.
    Chain-of-thought is forced before the JSON judgment.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model

    def verify_edit(
        self,
        original: Image.Image,
        result: Image.Image,
        instruction: str,
        target: str,
        verb: str = "remove",
    ) -> CriticVerdict:
        """Verify whether the edit instruction was fulfilled.

        Images are interleaved with labeled text parts so the model
        cannot confuse which is the original and which is the result.
        """
        orig_b64 = _image_to_base64(original)
        result_b64 = _image_to_base64(result)

        prompt_text = (
            f'You are a strict image editing judge. '
            f'Image 1 is the ORIGINAL. Image 2 is the RESULT. '
            f'Instruction: "{instruction}". Target: "{target}". Action: "{verb}". '
            f'Count ALL instances of "{target}" in the original. '
            f'Check if EACH one was edited in the result. '
            f'If instruction says all/every and ANY is missed, fulfilled=false and score<0.7. '
            f'Respond with JSON only.'
        )

        # Single prompt text + two images
        parts = [
            {"text": prompt_text},
            {"inline_data": {"mime_type": "image/png", "data": orig_b64}},
            {"inline_data": {"mime_type": "image/png", "data": result_b64}},
        ]

        response_schema = {
            "type": "OBJECT",
            "properties": {
                "fulfilled": {"type": "BOOLEAN"},
                "confidence": {"type": "NUMBER"},
                "semantic_score": {"type": "NUMBER"},
                "reasoning": {"type": "STRING"},
                "residual_objects": {"type": "ARRAY", "items": {"type": "STRING"}},
                "suggestions": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["fulfilled", "confidence", "semantic_score", "reasoning",
                         "residual_objects", "suggestions"],
        }

        body = json.dumps({
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
                "temperature": 0.1,
                "maxOutputTokens": 2048,
            },
        }).encode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{parse.quote(self.model, safe='')}:generateContent"
        )
        req = request.Request(url, data=body, method="POST",
                              headers={
                                  "Content-Type": "application/json; charset=utf-8",
                                  "x-goog-api-key": self.api_key,
                              })

        try:
            with request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No candidates in response")
            parts_out = candidates[0].get("content", {}).get("parts", [])
            response_text = "".join(p.get("text", "") for p in parts_out)

            logger.info("Raw response: %s", response_text[:500])
            return self._parse_response(response_text, instruction, target)
        except Exception as exc:
            logger.error("VLM Critic error: %s", exc)
            return CriticVerdict(
                fulfilled=False,  # Error = NOT approved, trigger retry
                confidence=0.0,
                semantic_score=0.3,
                reasoning=f"VLM Critic unavailable: {exc}",
                residual_objects=[],
                suggestions=["VLM Critic failed — retry with fresh request"],
            )

    def narrate_step(
        self,
        image: Image.Image,
        tool_name: str,
        step_description: str,
    ) -> str:
        """Generate verbose narration of what's happening at this step."""
        prompt = (
            f"You are an image editing assistant narrating an editing session.\n"
            f"The tool '{tool_name}' was just called: {step_description}\n"
            f"Look at the current image and describe:\n"
            f"1. What you see\n2. Why this tool was chosen\n3. Expected result\n"
            f"Be concise. 2-3 sentences."
        )
        img_b64 = _image_to_base64(image)
        try:
            return _call_gemini(self.api_key, self.model, prompt, [img_b64], 0.3).strip()
        except Exception as exc:
            return f"(Narration unavailable: {exc})"

    # _build_preamble and _build_analysis_prompt are no longer needed
    # — the prompt is now inline in verify_edit for simplicity

    def _parse_response(self, text: str, instruction: str, target: str) -> CriticVerdict:
        # Try 1: Direct JSON parse (if responseMimeType=application/json worked)
        try:
            data = json.loads(text.strip())
            return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try 2: Extract JSON from mixed text+JSON response
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Try 3: Heuristic extraction from raw text
        logger.warning("VLM Critic parse failed, raw: %s", text[:300])
        # Look for score-like patterns
        score = 0.4
        score_match = re.search(r'semantic_score["\s:]+([\d.]+)', text)
        if score_match:
            score = float(score_match.group(1))
        fulfilled = score >= 0.7

        return CriticVerdict(
            fulfilled=fulfilled,
            confidence=0.3,
            semantic_score=score,
            reasoning=text[:500] if text else "Parse error",
            residual_objects=[],
            suggestions=["VLM response was not JSON — using heuristic extraction"],
        )

    def _verdict_from_dict(self, data: dict) -> CriticVerdict:
        return CriticVerdict(
            fulfilled=bool(data.get("fulfilled", False)),
            confidence=float(data.get("confidence", 0.5)),
            semantic_score=float(data.get("semantic_score", 0.5)),
            reasoning=str(data.get("reasoning", "No reasoning provided")),
            residual_objects=list(data.get("residual_objects", [])),
            suggestions=list(data.get("suggestions", [])),
        )


class OllamaVLMCritic:
    """Uses Ollama locally as a VLM to verify edits semantically.

    Compatible with llama3.2-vision or llava models running via Ollama.
    """

    def __init__(self, model: str = "llama3.2-vision", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def verify_edit(
        self,
        original: Image.Image,
        result: Image.Image,
        instruction: str,
        target: str,
        verb: str = "remove",
    ) -> CriticVerdict:
        """Verify the edit by querying a local Ollama vision model."""
        orig_b64 = _image_to_base64(original)
        result_b64 = _image_to_base64(result)

        prompt_text = (
            f"You are a strict image editing judge. "
            f"Image 1 is the ORIGINAL. Image 2 is the RESULT. "
            f"Instruction: '{instruction}'. Target: '{target}'. Action: '{verb}'. "
            f"Count ALL instances of '{target}' in the original, and check if EACH one "
            f"was edited in the result. If the instruction says all/every and ANY is missed, "
            f"fulfilled must be false and score must be < 0.7. "
            f"Return ONLY valid JSON matching this structure: "
            f"{{\"fulfilled\": true/false, \"confidence\": 0.0-1.0, \"semantic_score\": 0.0-1.0, "
            f"\"reasoning\": \"string\", \"residual_objects\": [\"string\"], \"suggestions\": [\"string\"]}}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [orig_b64, result_b64]
                }
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }

        url = f"{self.host}/api/chat"
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                              headers={"Content-Type": "application/json"})

        try:
            with request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            response_text = data.get("message", {}).get("content", "")
            logger.info("Ollama raw response: %s", response_text[:500])
            return self._parse_response(response_text, instruction, target)
        except Exception as exc:
            logger.error("Ollama VLM error: %s", exc)
            return CriticVerdict(
                fulfilled=False,
                confidence=0.0,
                semantic_score=0.3,
                reasoning=f"Ollama Critic unavailable: {exc}",
                residual_objects=[],
                suggestions=["Ollama connection failed — retry or start ollama serve"],
            )

    def _parse_response(self, text: str, instruction: str, target: str) -> CriticVerdict:
        try:
            data = json.loads(text.strip())
            return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback 1: Extract JSON block
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback 2: Heuristic extraction
        score = 0.4
        score_match = re.search(r'"semantic_score"[\s:]+([\d.]+)', text)
        if score_match:
            score = float(score_match.group(1))
        
        fulfilled_match = re.search(r'"fulfilled"[\s:]+(true|false)', text, re.IGNORECASE)
        fulfilled = score >= 0.7
        if fulfilled_match:
            fulfilled = fulfilled_match.group(1).lower() == 'true'

        return CriticVerdict(
            fulfilled=fulfilled,
            confidence=0.3,
            semantic_score=score,
            reasoning=text[:500] if text else "Parse error from Ollama",
            residual_objects=[],
            suggestions=["Could not fully parse Ollama json response"],
        )

    def _verdict_from_dict(self, data: dict) -> CriticVerdict:
        return CriticVerdict(
            fulfilled=bool(data.get("fulfilled", False)),
            confidence=float(data.get("confidence", 0.5)),
            semantic_score=float(data.get("semantic_score", 0.5)),
            reasoning=str(data.get("reasoning", "No reasoning provided")),
            residual_objects=list(data.get("residual_objects", [])),
            suggestions=list(data.get("suggestions", [])),
        )


class HuggingFaceVLMCritic:
    """Uses Hugging Face Serverless Inference API as a VLM to verify edits.

    Specifically uses the OpenAI-compatible /v1/chat/completions endpoint
    for models like meta-llama/Llama-3.2-11B-Vision-Instruct.
    """

    def __init__(
        self,
        api_token: str,
        model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    ):
        self.api_token = api_token
        self.model = model
        self.base_url = f"https://router.huggingface.co/hf-inference/models/{self.model}/v1/chat/completions"

    def verify_edit(
        self,
        original: Image.Image,
        result: Image.Image,
        instruction: str,
        target: str,
        verb: str = "remove",
    ) -> CriticVerdict:
        """Verify the edit by querying Hugging Face Inference API."""
        # Convert to lower resolution JPEG for HF payload limits
        orig_img = original.copy()
        orig_img.thumbnail((512, 512))
        res_img = result.copy()
        res_img.thumbnail((512, 512))
        
        orig_b64 = _image_to_base64(orig_img)
        result_b64 = _image_to_base64(res_img)

        prompt_text = (
            f"You are a strict image editing judge. "
            f"Image 1 is the ORIGINAL. Image 2 is the RESULT. "
            f"Instruction: '{instruction}'. Target: '{target}'. Action: '{verb}'. "
            f"Count ALL instances of '{target}' in the original, and check if EACH one "
            f"was edited in the result. If the instruction says all/every and ANY is missed, "
            f"fulfilled must be false and score must be < 0.7. "
            f"Return ONLY valid JSON matching this structure exactly (no markdown): "
            f"{{\"fulfilled\": true, \"confidence\": 0.9, \"semantic_score\": 0.9, "
            f"\"reasoning\": \"string\", \"residual_objects\": [\"string\"], \"suggestions\": [\"string\"]}}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{result_b64}"}}
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,
            "stream": False
        }

        req = request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_token}",
            }
        )

        try:
            with request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            choices = data.get("choices", [])
            if not choices:
                raise ValueError("No choices in response")
            
            response_text = choices[0].get("message", {}).get("content", "")
            logger.info("HF VLM raw response: %s", response_text[:500])
            return self._parse_response(response_text, instruction, target)
        except error.HTTPError as e:
            err_body = e.read().decode('utf-8')
            logger.error("HF VLM API error: %d - %s", e.code, err_body[:500])
            error_msg = f"Hugging Face API Error {e.code}"
            return self._fallback_error(error_msg)
        except Exception as exc:
            logger.error("HF VLM error: %s", exc)
            return self._fallback_error(f"HF Critic unavailable: {exc}")

    def _fallback_error(self, msg: str) -> CriticVerdict:
        return CriticVerdict(
            fulfilled=False,
            confidence=0.0,
            semantic_score=0.3,
            reasoning=msg,
            residual_objects=[],
            suggestions=["Hugging Face connection failed — check API token and tier"],
        )

    def _parse_response(self, text: str, instruction: str, target: str) -> CriticVerdict:
        text = text.strip()
        
        # 1. Direct JSON attempt
        try:
            data = json.loads(text)
            return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Extract JSON block (```json ... ```)
        try:
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
            if json_match:
                data = json.loads(json_match.group(1))
                return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # 3. Extract any curly braces block
        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                return self._verdict_from_dict(data)
        except (json.JSONDecodeError, ValueError):
            pass

        # 4. Fallback Regex Heuristic
        score = 0.4
        score_match = re.search(r'"semantic_score"[\s:]+([\d.]+)', text)
        if score_match:
            score = float(score_match.group(1))
        
        fulfilled_match = re.search(r'"fulfilled"[\s:]+(true|false)', text, re.IGNORECASE)
        fulfilled = score >= 0.7
        if fulfilled_match:
            fulfilled = fulfilled_match.group(1).lower() == 'true'

        return CriticVerdict(
            fulfilled=fulfilled,
            confidence=0.3,
            semantic_score=score,
            reasoning=text[:500] if text else "Parse error from HF VLM",
            residual_objects=[],
            suggestions=["Could not firmly parse JSON from HF model response"],
        )

    def _verdict_from_dict(self, data: dict) -> CriticVerdict:
        return CriticVerdict(
            fulfilled=bool(data.get("fulfilled", False)),
            confidence=float(data.get("confidence", 0.5)),
            semantic_score=float(data.get("semantic_score", 0.5)),
            reasoning=str(data.get("reasoning", "No reasoning provided")),
            residual_objects=list(data.get("residual_objects", [])),
            suggestions=list(data.get("suggestions", [])),
        )


__all__ = ["VLMCritic", "OllamaVLMCritic", "HuggingFaceVLMCritic", "CriticVerdict"]
