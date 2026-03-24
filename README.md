# Agent Banana

Agent Banana is an end-to-end image editing agent with explicit planning, preview-first localization, Florence-2 grounding, region editing, and quality gating.

## What it does

- Parses a natural-language edit request into atomic edit steps.
- Enumerates candidate edit paths at planning time and ranks them with a persisted RL-style value store.
- Generates one Nano Banana preview before each bounding-box decision.
- Uses Florence-2 phrase grounding on the source image to localize the target object instead of pixel-diff bbox heuristics.
- Tightens the grounded region with target-aware priors for small accessories like glasses.
- Edits only the selected crop, composites it back into the original image, and rejects oversized or overly destructive local edits.
- Persists session memory and planner feedback across turns.

## Quick start

Install the package:

```bash
python -m pip install -e .
```

The first live run downloads the grounding model weights. Use `florence-community/Florence-2-base` by default, or override it with `AGENT_BANANA_GROUNDING_MODEL`.

Run the browser UI:

```bash
python -m agent_banana.server --host 127.0.0.1 --port 8010
```

Or run the CLI:

```bash
python -m agent_banana.cli \
  --image /path/to/input.png \
  --instruction "Remove only the glasses from the woman's face. Preserve her eyes, skin, hair, and pose."
```

## Environment

If `GEMINI_API_KEY` or `GOOGLE_API_KEY` is set, the app uses Gemini image generation for preview creation and crop editing.

Bounding-box localization uses Florence-2 through `transformers` and `torch`. If the model cannot be loaded, the app falls back to a deterministic mock localizer so tests and UI development still run end to end.

Without an API key, the app still runs end to end in deterministic mock mode so the planner, bbox logic, quality gates, and UI remain testable.

Optional `.env`:

```bash
GEMINI_API_KEY="replace-with-your-gemini-api-key"
AGENT_BANANA_IMAGE_MODEL="gemini-2.5-flash-image"
AGENT_BANANA_GROUNDING_MODEL="florence-community/Florence-2-base"
```

## Project layout

- `src/agent_banana/planning.py`: instruction parsing, path search, and RL-style scoring.
- `src/agent_banana/nano_banana.py`: Gemini Nano Banana client and deterministic mock backend.
- `src/agent_banana/vlm_localizer.py`: Florence-2 phrase grounding and mock localizer fallback.
- `src/agent_banana/pipeline.py`: orchestration across planning, previewing, VLM localization, editing, compositing, and reward updates.
- `src/agent_banana/targeting.py`: target-aware priors for compact edits such as glasses removal.
- `src/agent_banana/quality.py`: local quality gate with size-aware rejection.
- `src/agent_banana/server.py`: local browser UI.
- `tests/test_agent_banana.py`: regression coverage for planning, localization, and end-to-end flow.
