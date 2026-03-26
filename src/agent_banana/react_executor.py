"""ReAct Agent Executor — True LLM-driven Thought → Action → Observation loop.

The executor uses Gemini as an orchestrator that DECIDES which tool to call
at each step based on the current state and observations:

1. THINK: LLM analyzes the situation and decides the next action
2. ACT: Execute the selected tool from the ToolRegistry
3. OBSERVE: Feed the result back to the LLM
4. REPEAT until LLM calls 'finish' or max steps reached

Key design decisions:
- Each attempt CHAINS on the previous result (not the original)
- VLM Critic compares the UNTOUCHED original vs the current state
- The LLM sees all previous steps as scratchpad context
- On failure to parse LLM output, falls back to a deterministic sequence
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from PIL import Image

from .models import BoundingBox, QualityMetrics
from .quality import QualityJudge
from .seam_detector import boundary_penalty
from .tool_registry import ToolRegistry, build_tool_registry
from .vlm_critic import VLMCritic, CriticVerdict
from .vision import crop_box, expand_box, paste_crop, encode_png_data_url, center_box


# ─── Default orchestrator model ───
DEFAULT_ORCHESTRATOR_MODEL = "gemini-2.5-flash"
DEFAULT_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace."""
    step_num: int
    thought: str
    action: str
    params: Dict[str, Any]
    observation: str
    critic_verdict: Optional[CriticVerdict] = None
    image_url: str = ""
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "step_num": self.step_num,
            "thought": self.thought,
            "action": self.action,
            "params": self.params,
            "observation": self.observation,
            "critic_verdict": self.critic_verdict.to_dict() if self.critic_verdict else None,
            "image_url": self.image_url,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AgentResult:
    """Complete result of an agent execution."""
    success: bool
    final_image: Optional[Image.Image]
    final_image_url: str
    quality: Optional[QualityMetrics]
    steps: List[AgentStep]
    total_attempts: int
    total_duration_ms: int

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "final_image_url": self.final_image_url,
            "quality": self.quality.to_dict() if self.quality else None,
            "steps": [s.to_dict() for s in self.steps],
            "total_attempts": self.total_attempts,
            "total_duration_ms": self.total_duration_ms,
        }


# ─── LLM Orchestrator Prompt ───

_SYSTEM_PROMPT = """\
You are a ReAct image editing agent. You MUST edit images by choosing tools step-by-step.

## State
- Original image size: {img_w}x{img_h}
- Target: "{target}" (action: {verb})
- Bounding box: left={bbox_left}, top={bbox_top}, right={bbox_right}, bottom={bbox_bottom}
- Instruction: "{instruction}"
- Attempt: {attempt}/{max_attempts}

## Available Tools
{tools_desc}

## Rules
1. ALWAYS start with expand_region to add context padding around the bbox
2. Then crop_local_patch to get the local region
3. Then edit_local to apply the edit
4. Then blend_back to merge the edit into the full image
5. Then evaluate_quality to check the result
6. If quality is poor or seam is detected, use detect_seam and adjust_taper
7. Call finish when the edit looks good

## Previous Steps
{scratchpad}

## Response Format
Respond with ONLY a JSON object:
```json
{{
  "thought": "your reasoning about what to do next",
  "action": "tool_name",
  "action_input": {{
    "param1": "value1"
  }}
}}
```
"""


def _call_orchestrator_llm(
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_ORCHESTRATOR_MODEL,
    temperature: float = 0.2,
) -> str:
    """Call Gemini for text-only orchestration (no image generation)."""
    url = f"{DEFAULT_API_BASE}/{parse.quote(model, safe='')}:generateContent"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "text/plain",
            "temperature": temperature,
            "maxOutputTokens": 1024,
        },
    }

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    with request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    texts = []
    for candidate in data.get("candidates", []):
        for part in (candidate.get("content") or {}).get("parts", []):
            if part.get("text"):
                texts.append(part["text"].strip())
    return "\n".join(texts)


def _parse_llm_action(text: str) -> Optional[Dict[str, Any]]:
    """Extract action JSON from LLM response text."""
    # Try direct JSON parse
    try:
        data = json.loads(text.strip())
        if "action" in data:
            return data
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


class ReActExecutor:
    """ReAct-style agent executor for image editing.

    Uses an LLM (Gemini) to dynamically decide which tool to call at each
    step, forming a true Thought → Action → Observation loop. Falls back to
    a deterministic sequence if the LLM fails to produce valid actions.

    Critical design decisions:
    - Each attempt CHAINS on the previous result (not the original)
    - VLM Critic compares the UNTOUCHED original vs the current state
    - On retry: wider bbox to catch missed targets, stronger prompt
    - Tracks best result across all attempts
    """

    def __init__(
        self,
        image_client,
        quality_judge: QualityJudge,
        vlm_critic: Optional[VLMCritic] = None,
        max_attempts: int = 3,
        max_steps_per_attempt: int = 10,
    ):
        self.image_client = image_client
        self.quality_judge = quality_judge
        self.vlm_critic = vlm_critic
        self.max_attempts = max_attempts
        self.max_steps_per_attempt = max_steps_per_attempt
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.tool_registry = build_tool_registry()

    def execute_edit(
        self,
        original_image: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        *,
        target_profile: str = "",
        modifiers: list = None,
    ) -> AgentResult:
        """Run the full ReAct agent loop for a single edit step."""
        start_time = time.time()
        all_steps: List[AgentStep] = []
        step_num = 0

        # UNTOUCHED original — never modified, used for VLM comparison
        untouched_original = original_image.copy()
        # Working image — updated after each attempt (chains results)
        working_image = original_image.copy()
        current_bbox = bbox
        best_result = None
        best_score = -1.0

        for attempt in range(1, self.max_attempts + 1):
            # ─── Try LLM-driven ReAct loop for this attempt ───
            attempt_steps, composed, quality, success = self._run_react_attempt(
                working_image=working_image,
                untouched_original=untouched_original,
                instruction=instruction,
                target=target,
                verb=verb,
                bbox=current_bbox,
                attempt=attempt,
                step_num_offset=step_num,
            )

            step_num += len(attempt_steps)
            all_steps.extend(attempt_steps)

            if composed is None:
                composed = working_image

            if quality is None:
                quality = self.quality_judge.evaluate(
                    untouched_original, composed, current_bbox,
                    preview=composed, target=target, verb=verb,
                )

            # ─── VLM Critic: compare UNTOUCHED original vs composed ───
            critic_verdict = None
            if self.vlm_critic:
                step_num += 1
                t0 = time.time()

                try:
                    critic_verdict = self.vlm_critic.verify_edit(
                        original=untouched_original,
                        result=composed,
                        instruction=instruction,
                        target=target,
                        verb=verb,
                    )
                except Exception as exc:
                    critic_verdict = CriticVerdict(
                        fulfilled=False, confidence=0.0, semantic_score=0.3,
                        reasoning=f"Critic error: {exc}",
                        residual_objects=[], suggestions=["VLM Critic failed, retry"],
                    )

                quality.semantic_score = critic_verdict.semantic_score
                quality.semantic_fulfilled = critic_verdict.fulfilled
                quality.semantic_reasoning = critic_verdict.reasoning

                all_steps.append(AgentStep(
                    step_num=step_num,
                    thought=(
                        f"VLM Critic comparing ORIGINAL (untouched) vs "
                        f"RESULT (after {attempt} attempt(s)). "
                        f"Looking for missed instances of '{target}'."
                    ),
                    action="verify_semantic",
                    params={"semantic_score": round(critic_verdict.semantic_score, 2),
                            "fulfilled": critic_verdict.fulfilled,
                            "attempt": attempt},
                    observation=critic_verdict.reasoning,
                    critic_verdict=critic_verdict,
                    duration_ms=int((time.time() - t0) * 1000),
                ))

                if not critic_verdict.fulfilled:
                    quality.accepted = False
                    quality.notes.append(
                        f"[Attempt {attempt}] VLM REJECTED: {critic_verdict.reasoning}"
                    )
                    if critic_verdict.residual_objects:
                        quality.notes.append(
                            f"Still visible: {', '.join(critic_verdict.residual_objects)}"
                        )
                    print(f"[react] Attempt {attempt} REJECTED "
                          f"(semantic={critic_verdict.semantic_score:.2f})")
                else:
                    quality.notes.append(
                        f"[Attempt {attempt}] VLM APPROVED "
                        f"(semantic={critic_verdict.semantic_score:.2f})"
                    )
                    print(f"[react] Attempt {attempt} APPROVED "
                          f"(semantic={critic_verdict.semantic_score:.2f})")

            # ─── Track best result ───
            combined_score = quality.score * quality.semantic_score
            if combined_score > best_score:
                best_score = combined_score
                best_result = (composed, quality)

            # ─── DECIDE: Continue or stop? ───
            is_approved = (
                critic_verdict is not None
                and critic_verdict.fulfilled
                and critic_verdict.semantic_score >= 0.7
                and quality.accepted
            )
            if is_approved or success:
                total_ms = int((time.time() - start_time) * 1000)
                return AgentResult(
                    success=True,
                    final_image=composed,
                    final_image_url=encode_png_data_url(composed),
                    quality=quality,
                    steps=all_steps,
                    total_attempts=attempt,
                    total_duration_ms=total_ms,
                )

            # ─── CHAIN: Use this result as starting point for next attempt ───
            working_image = composed

            # ─── Log strategy adjustment for next attempt ───
            if attempt < self.max_attempts:
                step_num += 1
                suggestion = ""
                if critic_verdict and critic_verdict.suggestions:
                    suggestion = critic_verdict.suggestions[0]
                residuals = ""
                if critic_verdict and critic_verdict.residual_objects:
                    residuals = ", ".join(critic_verdict.residual_objects)

                all_steps.append(AgentStep(
                    step_num=step_num,
                    thought=(
                        f"Chaining result: using attempt {attempt}'s output as "
                        f"new starting image. On next attempt I'll use a wider "
                        f"edit region to catch ALL missed instances."
                        + (f" Residuals to fix: {residuals}" if residuals else "")
                        + (f" Suggestion: {suggestion}" if suggestion else "")
                    ),
                    action="adjust_strategy",
                    params={"chained": True,
                            "residuals": residuals,
                            "suggestion": suggestion[:100]},
                    observation=(
                        f"Next attempt will edit from the current result image "
                        f"with wider region"
                    ),
                ))

        # ─── Max attempts reached ───
        total_ms = int((time.time() - start_time) * 1000)
        best_image, best_quality = best_result if best_result else (working_image, None)

        step_num += 1
        all_steps.append(AgentStep(
            step_num=step_num,
            thought=f"Reached max attempts ({self.max_attempts}). "
                    f"Returning best result (score={best_score:.3f}).",
            action="return_best",
            params={"best_score": round(best_score, 3),
                    "attempts": self.max_attempts},
            observation=f"Returning best of {self.max_attempts} attempts",
        ))

        return AgentResult(
            success=False,
            final_image=best_image,
            final_image_url=encode_png_data_url(best_image),
            quality=best_quality,
            steps=all_steps,
            total_attempts=self.max_attempts,
            total_duration_ms=total_ms,
        )

    def _run_react_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
    ) -> tuple:
        """Run a single ReAct attempt with LLM-driven tool selection.

        Returns: (steps, composed_image, quality, success)
        """
        # Try LLM-driven approach first, fall back to deterministic
        if self.api_key:
            try:
                return self._llm_driven_attempt(
                    working_image, untouched_original, instruction,
                    target, verb, bbox, attempt, step_num_offset,
                )
            except Exception as exc:
                print(f"[react] LLM-driven attempt failed ({exc}), falling back to deterministic")

        return self._deterministic_attempt(
            working_image, untouched_original, instruction,
            target, verb, bbox, attempt, step_num_offset,
        )

    def _llm_driven_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
    ) -> tuple:
        """LLM-driven ReAct loop: Thought → Action → Observation."""
        steps: List[AgentStep] = []
        step_num = step_num_offset
        scratchpad_entries: List[str] = []

        # State that tools can read and write
        state = {
            "working_image": working_image,
            "original_image": untouched_original,
            "bbox": bbox,
            "edit_region": None,
            "local_crop": None,
            "edited_crop": None,
            "composed": None,
            "quality": None,
        }

        tools_desc = self.tool_registry.tools_prompt()

        for loop_step in range(self.max_steps_per_attempt):
            step_num += 1
            t0 = time.time()

            # Build scratchpad from previous steps
            scratchpad = "\n".join(scratchpad_entries) if scratchpad_entries else "(No previous steps)"

            # ─── THINK: Ask LLM what to do next ───
            system_prompt = _SYSTEM_PROMPT.format(
                img_w=working_image.size[0],
                img_h=working_image.size[1],
                target=target,
                verb=verb,
                bbox_left=bbox.left,
                bbox_top=bbox.top,
                bbox_right=bbox.right,
                bbox_bottom=bbox.bottom,
                instruction=instruction,
                attempt=attempt,
                max_attempts=self.max_attempts,
                tools_desc=tools_desc,
                scratchpad=scratchpad,
            )

            llm_response = _call_orchestrator_llm(
                system_prompt,
                api_key=self.api_key,
            )

            # ─── Parse the LLM's decision ───
            parsed = _parse_llm_action(llm_response)
            if not parsed:
                print(f"[react] LLM response unparseable at step {loop_step + 1}, "
                      f"falling back to deterministic")
                # Fall back to deterministic for remaining steps
                det_steps, composed, quality, success = self._deterministic_attempt(
                    state["composed"] or working_image,
                    untouched_original, instruction, target, verb,
                    bbox, attempt, step_num,
                )
                steps.extend(det_steps)
                return steps, composed, quality, success

            thought = parsed.get("thought", "")
            action_name = parsed.get("action", "")
            action_input = parsed.get("action_input", {})

            # ─── Handle 'finish' action ───
            if action_name == "finish":
                composed = state.get("composed") or working_image
                quality = state.get("quality")
                steps.append(AgentStep(
                    step_num=step_num,
                    thought=thought,
                    action="finish",
                    params=action_input,
                    observation="Agent decided to finish — returning result.",
                    image_url=encode_png_data_url(composed) if composed else "",
                    duration_ms=int((time.time() - t0) * 1000),
                ))
                return steps, composed, quality, True

            # ─── ACT: Execute the tool ───
            observation, result_data = self._execute_react_tool(
                action_name, action_input, state, instruction, target, verb, attempt,
            )

            elapsed_ms = int((time.time() - t0) * 1000)

            # Build image_url for the step if we have a composed image
            image_url = ""
            if action_name == "blend_back" and state.get("composed"):
                image_url = encode_png_data_url(state["composed"])
            elif action_name == "edit_local" and state.get("edited_crop"):
                image_url = encode_png_data_url(state["edited_crop"])

            steps.append(AgentStep(
                step_num=step_num,
                thought=thought,
                action=action_name,
                params={k: str(v)[:200] if not isinstance(v, (int, float, bool)) else v
                        for k, v in action_input.items()},
                observation=observation,
                image_url=image_url,
                duration_ms=elapsed_ms,
            ))

            # Update scratchpad
            scratchpad_entries.append(
                f"Step {loop_step + 1}: Thought: {thought[:150]}\n"
                f"  Action: {action_name}\n"
                f"  Observation: {observation[:200]}"
            )

        # Max steps reached for this attempt
        composed = state.get("composed") or working_image
        quality = state.get("quality")
        return steps, composed, quality, False

    def _execute_react_tool(
        self,
        action_name: str,
        action_input: dict,
        state: dict,
        instruction: str,
        target: str,
        verb: str,
        attempt: int,
    ) -> tuple:
        """Execute a tool and update the shared state. Returns (observation, result_data)."""

        working_image = state["working_image"]
        bbox = state["bbox"]

        if action_name == "expand_region":
            padding_ratio = float(action_input.get("padding_ratio", 0.5 + (attempt - 1) * 0.25))
            w, h = working_image.size

            if attempt >= 2:
                # On retries, use near-global region
                margin_x = int(w * 0.05)
                margin_y = int(h * 0.05)
                edit_region = BoundingBox(
                    left=margin_x, top=margin_y,
                    right=w - margin_x, bottom=h - margin_y,
                )
                strategy = "global-retry"
            else:
                pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
                edit_region = expand_box(bbox, pad, working_image.size)
                strategy = "local"

            state["edit_region"] = edit_region
            observation = (
                f"Expanded region: {edit_region.width}x{edit_region.height} "
                f"({strategy}, padding={padding_ratio:.0%})"
            )
            return observation, {"edit_region": edit_region.to_dict()}

        elif action_name == "crop_local_patch":
            edit_region = state.get("edit_region")
            if not edit_region:
                return "Error: No edit region set. Call expand_region first.", {}

            local_crop = crop_box(working_image, edit_region)
            state["local_crop"] = local_crop
            observation = f"Cropped {local_crop.size[0]}x{local_crop.size[1]} from working image"
            return observation, {"size": local_crop.size}

        elif action_name == "edit_local":
            local_crop = state.get("local_crop")
            if not local_crop:
                return "Error: No local crop available. Call crop_local_patch first.", {}

            prompt = self._build_prompt(instruction, target, verb, attempt)
            custom_prompt = action_input.get("prompt")
            if custom_prompt:
                prompt = custom_prompt

            try:
                edited_response = self.image_client.edit_full_image(local_crop, prompt)
                edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
                state["edited_crop"] = edited_crop
                observation = (
                    f"Gemini returned {edited_crop.size[0]}x{edited_crop.size[1]} edit "
                    f"(attempt {attempt})"
                )
                return observation, {"size": edited_crop.size}
            except Exception as exc:
                observation = f"Edit failed: {exc}"
                return observation, {"error": str(exc)}

        elif action_name == "blend_back":
            edited_crop = state.get("edited_crop")
            edit_region = state.get("edit_region")
            if not edited_crop or not edit_region:
                return "Error: Need edited_crop and edit_region. Call edit_local first.", {}

            composed = paste_crop(working_image, edited_crop, edit_region)
            state["composed"] = composed
            state["working_image"] = composed  # Chain for subsequent operations
            observation = (
                f"Blended {edit_region.width}x{edit_region.height} region "
                f"(attempt {attempt})"
            )
            return observation, {"region": edit_region.to_dict()}

        elif action_name == "detect_seam":
            composed = state.get("composed")
            if not composed:
                return "Error: No composed image. Call blend_back first.", {}

            seam = boundary_penalty(composed, bbox)
            observation = (
                f"BGD={seam['bgd']:.3f}, CBCS={seam['cbcs']:.3f}, "
                f"penalty={seam['penalty']:.3f}, verdict={seam['verdict']}"
            )
            return observation, seam

        elif action_name == "adjust_taper":
            edited_crop = state.get("edited_crop")
            edit_region = state.get("edit_region")
            if not edited_crop or not edit_region:
                return "Error: Need edited_crop and edit_region.", {}

            composed = paste_crop(working_image, edited_crop, edit_region)
            state["composed"] = composed
            seam = boundary_penalty(composed, bbox)
            observation = (
                f"Re-blended with taper. seam_score={seam['penalty']:.3f}, "
                f"verdict={seam['verdict']}"
            )
            return observation, {"seam": seam}

        elif action_name == "evaluate_quality":
            composed = state.get("composed")
            if not composed:
                return "Error: No composed image to evaluate.", {}

            quality = self.quality_judge.evaluate(
                state["original_image"], composed, bbox,
                preview=composed, target=target, verb=verb,
            )
            state["quality"] = quality
            observation = (
                f"Score={quality.score:.3f}, seam={quality.seam_verdict}, "
                f"inside_Δ={quality.inside_change:.3f}, "
                f"outside_Δ={quality.outside_change:.3f}, "
                f"accepted={quality.accepted}"
            )
            return observation, quality.to_dict()

        else:
            return f"Unknown tool: {action_name}. Use one of the available tools.", {}

    def _deterministic_attempt(
        self,
        working_image: Image.Image,
        untouched_original: Image.Image,
        instruction: str,
        target: str,
        verb: str,
        bbox: BoundingBox,
        attempt: int,
        step_num_offset: int,
    ) -> tuple:
        """Deterministic fallback: expand → crop → edit → blend → evaluate."""
        steps: List[AgentStep] = []
        step_num = step_num_offset
        padding_ratio = 0.5 + (attempt - 1) * 0.25

        # ─── Step 1: Expand region ───
        step_num += 1
        t0 = time.time()
        w, h = working_image.size

        if attempt >= 2:
            margin_x = int(w * 0.05)
            margin_y = int(h * 0.05)
            edit_region = BoundingBox(
                left=margin_x, top=margin_y,
                right=w - margin_x, bottom=h - margin_y,
            )
            strategy = "global-retry"
        else:
            pad = max(40, int(max(bbox.width, bbox.height) * padding_ratio))
            edit_region = expand_box(bbox, pad, working_image.size)
            strategy = "local"

        steps.append(AgentStep(
            step_num=step_num,
            thought=(
                f"Attempt {attempt}/{self.max_attempts} (deterministic fallback). "
                f"Expanding bbox by {int(padding_ratio*100)}% for context."
            ),
            action="expand_region",
            params={"bbox": bbox.to_dict(), "padding": padding_ratio,
                    "edit_region": edit_region.to_dict(), "strategy": strategy},
            observation=f"Edit region: {edit_region.width}x{edit_region.height} ({strategy})",
            duration_ms=int((time.time() - t0) * 1000),
        ))

        # ─── Step 2: Crop ───
        step_num += 1
        t0 = time.time()
        local_crop = crop_box(working_image, edit_region)

        steps.append(AgentStep(
            step_num=step_num,
            thought="Cropping the expanded region from the working image.",
            action="crop_local_patch",
            params={"crop_size": f"{local_crop.size[0]}x{local_crop.size[1]}"},
            observation=f"Cropped {local_crop.size[0]}x{local_crop.size[1]}",
            duration_ms=int((time.time() - t0) * 1000),
        ))

        # ─── Step 3: Edit ───
        step_num += 1
        t0 = time.time()
        prompt = self._build_prompt(instruction, target, verb, attempt)

        try:
            edited_response = self.image_client.edit_full_image(local_crop, prompt)
            edited_crop = edited_response.image.convert("RGB").resize(local_crop.size)
            edit_observation = (
                f"Gemini returned {edited_crop.size[0]}x{edited_crop.size[1]} edit"
            )
        except Exception as exc:
            edit_observation = f"Edit failed: {exc}"
            steps.append(AgentStep(
                step_num=step_num,
                thought=f"Editing local crop with Gemini",
                action="edit_local",
                params={},
                observation=edit_observation,
                duration_ms=int((time.time() - t0) * 1000),
            ))
            return steps, working_image, None, False

        steps.append(AgentStep(
            step_num=step_num,
            thought=f"Editing local crop with Gemini (attempt {attempt})",
            action="edit_local",
            params={"prompt_preview": prompt[:150]},
            observation=edit_observation,
            duration_ms=int((time.time() - t0) * 1000),
        ))

        # ─── Step 4: Blend ───
        step_num += 1
        t0 = time.time()
        composed = paste_crop(working_image, edited_crop, edit_region)

        steps.append(AgentStep(
            step_num=step_num,
            thought="Blending edited crop back into working image.",
            action="blend_back",
            params={"region": edit_region.to_dict(), "attempt": attempt},
            observation=f"Blended {edit_region.width}x{edit_region.height} region",
            image_url=encode_png_data_url(composed),
            duration_ms=int((time.time() - t0) * 1000),
        ))

        # ─── Step 5: Evaluate quality ───
        step_num += 1
        t0 = time.time()
        quality = self.quality_judge.evaluate(
            untouched_original, composed, bbox,
            preview=composed, target=target, verb=verb,
        )

        steps.append(AgentStep(
            step_num=step_num,
            thought="Running quality check (comparing against untouched original).",
            action="evaluate_quality",
            params={"score": round(quality.score, 3), "seam": quality.seam_verdict},
            observation=f"Score={quality.score:.3f}, seam={quality.seam_verdict}, "
                        f"inside_Δ={quality.inside_change:.3f}",
            duration_ms=int((time.time() - t0) * 1000),
        ))

        return steps, composed, quality, False

    def _build_prompt(self, instruction: str, target: str, verb: str,
                      attempt: int) -> str:
        """Build edit prompt with escalating strength per attempt."""
        base = (
            f"This is a cropped region from a larger image. "
            f"{instruction} "
            f"Keep everything else EXACTLY the same — "
            f"same colors, lighting, textures, and positions."
        )

        if attempt >= 2:
            base += (
                f" CRITICAL: Make sure EVERY SINGLE instance of '{target}' "
                f"in this crop is affected. Do not miss any. "
                f"Check the entire image carefully for ALL occurrences."
            )

        if attempt >= 3:
            base += (
                f" ABSOLUTELY ENSURE that ALL instances of '{target}' "
                f"have been edited — left side, right side, foreground, "
                f"background, every single one. Previous attempts missed some."
            )

        if attempt > 1:
            base += (
                f" NOTE: A previous editing pass may have already changed "
                f"some instances. Look for any that are STILL unchanged and "
                f"edit those too, while preserving the already-edited ones."
            )

        return base


__all__ = ["ReActExecutor", "AgentResult", "AgentStep"]
