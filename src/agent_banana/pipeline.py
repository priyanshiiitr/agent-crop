from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

from .config import load_dotenv

from .memory import ContextFolder, SessionStore
from .models import PipelineResult, PlanStep, SessionState, StepResult, TurnRecord
from .nano_banana import MockNanoBananaClient, NanoBananaClient, build_image_client
from .planning import RLPlanner, RLValueStore
from .quality import QualityJudge
from .targeting import (
    classify_target,
    fallback_box_for_profile,
    grounding_phrases_for_target,
    rank_grounding_candidates,
    refine_bbox_for_profile,
)
from .vlm_localizer import MockVlmLocalizer, VlmLocalizer, build_localizer
from .vlm_localizer import GroundingResult
from .vision import (
    assess_preview_framing,
    center_box,
    crop_box,
    draw_bbox_overlay,
    encode_png_data_url,
    expand_box,
    fit_image_inside_canvas,
    paste_crop,
)


class AgentBananaApp:
    def __init__(
        self,
        *,
        root: Path,
        image_client: NanoBananaClient | None = None,
        localizer: VlmLocalizer | None = None,
        max_retries: int = 1,
    ):
        self.root = root
        artifacts_root = self.root / "artifacts" / "agent_banana"
        self.image_client = image_client or build_image_client()
        self.fallback_image_client = MockNanoBananaClient()
        self.localizer = localizer or build_localizer()
        self.fallback_localizer = MockVlmLocalizer()
        self.context_folder = ContextFolder()
        self.session_store = SessionStore(artifacts_root / "sessions")
        self.planner = RLPlanner(RLValueStore(artifacts_root / "planner_values.json"))
        self.quality_judge = QualityJudge()
        self.max_retries = max_retries

    @classmethod
    def from_env(cls, root: Path | None = None) -> "AgentBananaApp":
        root = root or Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        return cls(root=root)

    def run(self, image: Image.Image, instruction: str, session_id: str | None = None) -> PipelineResult:
        session = self.session_store.load_or_create(session_id)
        current_image = image.convert("RGB")
        folded_context = self.context_folder.fold(session.turns)
        parsed_edits = self.planner.parse_instruction(instruction, folded_context)
        edits_by_id = {edit.edit_id: edit for edit in parsed_edits}
        candidate_plans = self.planner.plan(parsed_edits, folded_context)
        selected_plan = candidate_plans[0]

        runtime_mode = self.image_client.mode_label()
        grounding_mode = self.localizer.mode_label()
        step_results = []
        reward_components = []
        bboxes = []

        for step in selected_plan.steps:
            edit = edits_by_id[step.edit_id]
            target_profile = classify_target(step.target, step.verb)
            preview_prompt = self._preview_prompt(instruction, step, folded_context)
            preview_response, preview_mode = self._safe_preview(current_image, preview_prompt)
            if preview_mode != runtime_mode:
                runtime_mode = preview_mode

            normalized_preview = self._prepare_preview_for_display(
                current_image,
                preview_response.image,
            )
            grounding_result = GroundingResult(phrases=[], candidates=[])
            ranked_candidates = []
            localizer_mode = grounding_mode
            if step.scope == "global":
                bbox = center_box(current_image.size, scale=0.82)
            else:
                grounding_phrases = grounding_phrases_for_target(step.target, edit.modifiers, step.verb)
                grounding_result, localizer_mode = self._safe_localize(current_image, grounding_phrases, target_profile)
                if localizer_mode != grounding_mode:
                    grounding_mode = localizer_mode
                ranked_candidates = rank_grounding_candidates(grounding_result.candidates, current_image.size, target_profile)
                if ranked_candidates:
                    bbox = refine_bbox_for_profile(ranked_candidates[0].bbox, current_image.size, target_profile)
                else:
                    bbox = fallback_box_for_profile(current_image.size, target_profile)

            composed_image, bbox, quality, attempts, edit_mode = self._apply_step(
                current_image,
                step,
                bbox,
                normalized_preview,
                folded_context.summary,
                target_profile,
            )
            if edit_mode != runtime_mode:
                runtime_mode = edit_mode

            overlay_image = draw_bbox_overlay(current_image, bbox, step.target)
            step_results.append(
                StepResult(
                    step=step,
                    bbox=bbox,
                    quality=quality,
                    preview_data_url=encode_png_data_url(normalized_preview),
                    overlay_data_url=encode_png_data_url(overlay_image),
                    edited_data_url=encode_png_data_url(composed_image),
                    attempts=attempts,
                    grounding_phrases=list(grounding_result.phrases),
                    grounding_candidates=ranked_candidates[:5],
                    localizer_mode=localizer_mode,
                )
            )
            reward_components.append(quality.score)
            bboxes.append(bbox)
            current_image = composed_image

        reward = 0.0 if not reward_components else sum(reward_components) / len(reward_components)
        self.planner.record_feedback(selected_plan, reward)
        session.turns.append(
            TurnRecord(
                instruction=instruction,
                parsed_edits=parsed_edits,
                selected_plan=selected_plan,
                reward=reward,
                bboxes=bboxes,
            )
        )
        session.folded_context = self.context_folder.fold(session.turns)
        self.session_store.save(session)

        return PipelineResult(
            session_id=session.session_id,
            mode=runtime_mode,
            grounding_mode=grounding_mode,
            instruction=instruction,
            folded_context=session.folded_context,
            parsed_edits=parsed_edits,
            candidate_plans=candidate_plans,
            selected_plan=selected_plan,
            source_image=encode_png_data_url(image),
            final_image=encode_png_data_url(current_image),
            step_results=step_results,
            reward=reward,
        )

    def _preview_prompt(self, instruction: str, step: PlanStep, folded_context) -> str:
        target_profile = classify_target(step.target, step.verb)
        profile_note = " Keep the exact same full-image framing, aspect ratio, camera position, and borders. Do not crop, zoom, or reframe the scene."
        if target_profile == "face_accessory" and step.verb in {"remove", "replace"}:
            profile_note += (
                " Localize only the eyewear region. Preserve the person's face, skin, eyes, nose, hair, pose, and lighting."
            )
        return (
            f"Session context: {folded_context.summary} "
            f"Current step {step.order}: {step.prompt} "
            f"Global user instruction: {instruction}.{profile_note}"
        )

    def _edit_prompt(self, step: PlanStep, context_summary: str, attempt: int) -> str:
        target_profile = classify_target(step.target, step.verb)
        retry_note = ""
        if attempt > 0:
            retry_note = " Retry with a slightly broader crop and stronger boundary consistency."
        profile_note = " Keep the crop framing fixed and do not zoom or change viewpoint."
        if target_profile == "face_accessory" and step.verb == "remove":
            profile_note += (
                " Remove only the glasses or frames. Keep the same face identity, eyes, eyebrows, nose bridge, skin tone, wrinkles, hair, and head pose unchanged. "
                "Inpaint only the pixels that were occluded by the glasses."
            )
        elif target_profile == "face_accessory" and step.verb == "replace":
            profile_note += (
                " Replace only the eyewear. Keep the same face identity, eyes, eyebrows, nose bridge, skin tone, wrinkles, hair, and head pose unchanged. "
                "Preserve the exact glasses placement while changing only the frame color and style requested."
            )
        elif target_profile == "small_accessory" and step.verb == "remove":
            profile_note += " Remove only the accessory and preserve the surrounding object or person."
        return f"Context: {context_summary} Step: {step.prompt}.{profile_note}{retry_note}"

    def _safe_preview(self, image: Image.Image, prompt: str):
        try:
            return self.image_client.generate_preview(image, prompt), self.image_client.mode_label()
        except Exception:
            return self.fallback_image_client.generate_preview(image, prompt), "mock-fallback"

    def _safe_edit(self, crop: Image.Image, prompt: str):
        try:
            return self.image_client.edit_crop(crop, prompt), self.image_client.mode_label()
        except Exception:
            return self.fallback_image_client.edit_crop(crop, prompt), "mock-fallback"

    def _safe_localize(self, image: Image.Image, phrases: list[str], profile: str):
        try:
            return self.localizer.localize(image, phrases, profile=profile), self.localizer.mode_label()
        except Exception:
            return self.fallback_localizer.localize(image, phrases, profile=profile), "mock-vlm-fallback"

    def _apply_step(
        self,
        current_image: Image.Image,
        step: PlanStep,
        bbox,
        preview_image: Image.Image,
        context_summary: str,
        target_profile: str,
    ) -> Tuple[Image.Image, object, object, int, str]:
        active_box = bbox
        runtime_mode = self.image_client.mode_label()

        for attempt in range(self.max_retries + 1):
            crop = crop_box(current_image, active_box)
            edit_prompt = self._edit_prompt(step, context_summary, attempt)
            edited_response, edit_mode = self._safe_edit(crop, edit_prompt)
            if edit_mode != runtime_mode:
                runtime_mode = edit_mode
            composed = paste_crop(current_image, edited_response.image, active_box)
            quality = self.quality_judge.evaluate(
                current_image,
                composed,
                active_box,
                preview=preview_image,
                target=step.target,
                verb=step.verb,
            )
            if quality.accepted or attempt == self.max_retries:
                return composed, active_box, quality, attempt + 1, runtime_mode
            if target_profile == "face_accessory":
                active_box = refine_bbox_for_profile(expand_box(active_box, 6, current_image.size), current_image.size, target_profile)
            else:
                active_box = expand_box(active_box, max(12, step.padding // 2), current_image.size)

        raise RuntimeError("Unreachable quality loop exit")

    def _prepare_preview_for_display(
        self,
        current_image: Image.Image,
        preview_image: Image.Image,
    ) -> Image.Image:
        normalized_preview = fit_image_inside_canvas(preview_image, current_image.size)
        assessment = assess_preview_framing(current_image, normalized_preview)
        if assessment["average"] > 0.10 or max(assessment["left"], assessment["right"]) > 0.14:
            return fit_image_inside_canvas(current_image, current_image.size)
        return normalized_preview
