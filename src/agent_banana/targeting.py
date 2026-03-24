from __future__ import annotations

import re

from .models import BoundingBox, GroundingCandidate

FACE_ACCESSORY_KEYWORDS = {"glasses", "eyeglasses", "spectacles", "sunglasses", "goggles", "frames", "eyewear"}
HEAD_ACCESSORY_KEYWORDS = {"hat", "cap", "helmet", "headband", "tiara", "veil"}
SMALL_ACCESSORY_KEYWORDS = {"earring", "earrings", "ring", "bracelet", "watch", "necklace", "pendant", "brooch"}
GLOBAL_KEYWORDS = {"background", "scene", "lighting", "style", "mood", "whole image", "entire image"}


def classify_target(target: str, verb: str = "") -> str:
    lowered = f"{verb} {target}".lower()
    if any(keyword in lowered for keyword in FACE_ACCESSORY_KEYWORDS):
        return "face_accessory"
    if any(keyword in lowered for keyword in HEAD_ACCESSORY_KEYWORDS):
        return "head_accessory"
    if any(keyword in lowered for keyword in SMALL_ACCESSORY_KEYWORDS):
        return "small_accessory"
    if any(keyword in lowered for keyword in GLOBAL_KEYWORDS):
        return "global_region"
    return "generic_local"


def grounding_phrases_for_target(target: str, modifiers: list[str], verb: str) -> list[str]:
    phrases: list[str] = []
    lowered_target = target.lower().strip()
    if lowered_target:
        phrases.append(lowered_target)

    if any(keyword in lowered_target for keyword in FACE_ACCESSORY_KEYWORDS):
        accessory_terms = [keyword for keyword in FACE_ACCESSORY_KEYWORDS if keyword in lowered_target]
        phrases.extend(accessory_terms)
        phrases.append("eyewear")

    if any(keyword in lowered_target for keyword in HEAD_ACCESSORY_KEYWORDS):
        phrases.extend(keyword for keyword in HEAD_ACCESSORY_KEYWORDS if keyword in lowered_target)

    if any(keyword in lowered_target for keyword in SMALL_ACCESSORY_KEYWORDS):
        phrases.extend(keyword for keyword in SMALL_ACCESSORY_KEYWORDS if keyword in lowered_target)

    if verb == "replace":
        cleaned_target = re.sub(r"\b(?:worn by|on|from|near)\b.*", "", lowered_target).strip()
        if cleaned_target:
            phrases.append(cleaned_target)

    for modifier in modifiers:
        if verb == "replace" and modifier.lower().startswith("with "):
            continue
        cleaned = modifier.lower().strip()
        if cleaned:
            phrases.append(cleaned)

    deduped = []
    seen = set()
    for phrase in phrases:
        normalized = " ".join(phrase.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def max_bbox_area_ratio(profile: str) -> float:
    return {
        "face_accessory": 0.07,
        "head_accessory": 0.14,
        "small_accessory": 0.05,
        "global_region": 1.0,
        "generic_local": 0.42,
    }.get(profile, 0.42)


def ideal_change_range(profile: str) -> tuple[float, float]:
    return {
        "face_accessory": (0.03, 0.24),
        "head_accessory": (0.04, 0.34),
        "small_accessory": (0.02, 0.22),
        "global_region": (0.05, 0.9),
        "generic_local": (0.03, 0.55),
    }.get(profile, (0.03, 0.55))


def fallback_box_for_profile(image_size: tuple[int, int], profile: str) -> BoundingBox:
    width, height = image_size
    if profile == "face_accessory":
        box_width = max(56, int(width * 0.22))
        box_height = max(28, int(height * 0.11))
        center_x = int(width * 0.38)
        center_y = int(height * 0.20)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    if profile == "head_accessory":
        box_width = max(72, int(width * 0.30))
        box_height = max(48, int(height * 0.17))
        center_x = int(width * 0.40)
        center_y = int(height * 0.13)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    if profile == "small_accessory":
        box_width = max(44, int(width * 0.16))
        box_height = max(44, int(height * 0.16))
        center_x = width // 2
        center_y = int(height * 0.42)
        return box_from_center(center_x, center_y, box_width, box_height, image_size)
    return box_from_center(width // 2, height // 2, max(64, int(width * 0.38)), max(64, int(height * 0.38)), image_size)


def rank_grounding_candidates(
    candidates: list[GroundingCandidate],
    image_size: tuple[int, int],
    profile: str,
) -> list[GroundingCandidate]:
    width, height = image_size
    image_area = max(1, width * height)

    def candidate_score(candidate: GroundingCandidate) -> float:
        area_ratio = candidate.bbox.area / image_area
        max_ratio = max_bbox_area_ratio(profile)
        size_score = 1.0 if area_ratio <= max_ratio else max(0.0, 1.0 - min(1.0, (area_ratio - max_ratio) / max_ratio))
        center_x = (candidate.bbox.left + candidate.bbox.right) / 2.0
        center_y = (candidate.bbox.top + candidate.bbox.bottom) / 2.0
        vertical_score = 1.0
        if profile in {"face_accessory", "head_accessory"}:
            target_band = 0.22 if profile == "face_accessory" else 0.18
            vertical_score = max(0.0, 1.0 - abs((center_y / max(1, height)) - target_band) / 0.35)
        horizontal_score = max(0.0, 1.0 - abs((center_x / max(1, width)) - 0.40) / 0.6)
        phrase_bonus = 0.1 if candidate.source == "phrase-grounding" else 0.0
        return 0.52 * candidate.score + 0.24 * size_score + 0.14 * vertical_score + 0.10 * horizontal_score + phrase_bonus

    return sorted(candidates, key=candidate_score, reverse=True)


def refine_bbox_for_profile(
    candidate: BoundingBox | None,
    image_size: tuple[int, int],
    profile: str,
) -> BoundingBox:
    width, height = image_size
    if candidate is None:
        return fallback_box_for_profile(image_size, profile)

    if profile == "face_accessory":
        max_width = max(56, int(width * 0.28))
        max_height = max(28, int(height * 0.14))
        center_x = (candidate.left + candidate.right) // 2
        center_y = min(int(height * 0.36), (candidate.top + candidate.bottom) // 2)
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    if profile == "head_accessory":
        max_width = max(72, int(width * 0.34))
        max_height = max(48, int(height * 0.20))
        center_x = (candidate.left + candidate.right) // 2
        center_y = min(int(height * 0.24), (candidate.top + candidate.bottom) // 2)
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    if profile == "small_accessory":
        max_width = max(44, int(width * 0.18))
        max_height = max(44, int(height * 0.18))
        center_x = (candidate.left + candidate.right) // 2
        center_y = (candidate.top + candidate.bottom) // 2
        refined_width = min(max_width, max(max_width // 2, candidate.width))
        refined_height = min(max_height, max(max_height // 2, candidate.height))
        return box_from_center(center_x, center_y, refined_width, refined_height, image_size)

    return candidate


def box_from_center(center_x: int, center_y: int, width: int, height: int, image_size: tuple[int, int]) -> BoundingBox:
    image_width, image_height = image_size
    half_width = max(1, width // 2)
    half_height = max(1, height // 2)
    left = max(0, center_x - half_width)
    top = max(0, center_y - half_height)
    right = min(image_width, center_x + half_width)
    bottom = min(image_height, center_y + half_height)
    return BoundingBox(left=left, top=top, right=right, bottom=bottom)
