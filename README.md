# 🍌 Moleculyst — Agentic Image Editing

An agentic image editing pipeline inspired by the [Agent Banana paper](https://arxiv.org/abs/2602.09084) (arXiv:2602.09084). Implements **Image Layer Decomposition (ILD)** for high-fidelity, localized edits with seamless blending.

## ✨ Features

- **Ground-First Local Inpainting** — Locates the target object *before* editing, crops a local patch, and sends it to Gemini for context-aware editing
- **Laplacian Pyramid Blending** — Multi-band blending (Burt & Adelson, 1983) seamlessly fuses edited patches back into the original
- **LLM Grounding Advisor** — Gemini 2.5 Flash reasons about spatial context and disambiguates targets (e.g., "glasses" → drinking glasses vs eyewear)
- **Interactive BBox Editor** — Draw custom bounding boxes on the original image to fine-tune the edit region
- **Custom Reconstruction Instructions** — Type specific instructions (e.g., "fill with table texture") for each recompose
- **Iterative Editing Loop** — Each output becomes the input for the next round — refine endlessly
- **Agentic Timeline UI** — Full transparency: reasoning, grounding phrases, spatial guidance, quality scores

## 🏗️ Architecture

```
User Instruction
    │
    ▼
┌──────────────────┐
│   LLM Planner    │ ── Parse instruction → plan steps
└────────┬─────────┘
         │
    ┌────▼────────────────────────────────────┐
    │           For each step:                │
    │                                         │
    │  1. GROUND (Florence-2 + LLM Advisor)   │
    │     └─ Find target on original image    │
    │                                         │
    │  2. CROP LOCAL PATCH                    │
    │     └─ bbox + 50% padding from original │
    │                                         │
    │  3. EDIT LOCALLY (Gemini)               │
    │     └─ Model sees surrounding context   │
    │     └─ Acts as inpainter                │
    │                                         │
    │  4. BLEND BACK (Laplacian Pyramid)      │
    │     └─ Multi-band frequency blending    │
    │     └─ Low-freq: wide color smoothing   │
    │     └─ High-freq: crisp edges           │
    └─────────────────────────────────────────┘
         │
         ▼
    Final Image (original pixels preserved outside edit region)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Gemini API key](https://aistudio.google.com/)

### Setup

```bash
# Clone and enter the project
cd agent-crop

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Configure API key
echo "GEMINI_API_KEY=your-key-here" > .env
```

### Run

```bash
python -m agent_banana.server --host 127.0.0.1 --port 8011
```

Open **http://127.0.0.1:8011** in your browser.

## 🎯 Usage

1. **Upload an image** and type an instruction (e.g., "remove the glasses from the table")
2. **Review** the agentic timeline: LLM reasoning → grounding → local edit → composition
3. **Adjust the bounding box** by drawing on the original image
4. **Type custom instructions** in the text field (e.g., "fill the area with wooden texture")
5. **Click Re-compose** — a new editor appears on the result for further refinement
6. **Iterate** until satisfied

## 📁 Project Structure

```
src/agent_banana/
├── server.py                 # Web UI + API endpoints
├── pipeline.py               # ILD pipeline: ground → crop → edit → blend
├── vision.py                 # Laplacian pyramid blending + image utilities
├── nano_banana.py            # Gemini API client
├── llm_grounding_advisor.py  # LLM spatial reasoning advisor
├── vlm_localizer.py          # Florence-2 grounding
├── targeting.py              # Target classification + bbox refinement
├── planning.py               # RL-based edit planner
├── quality.py                # Quality evaluation judge
├── models.py                 # Data models (BoundingBox, StepResult, etc.)
├── memory.py                 # Context folding + session storage
└── config.py                 # Environment configuration
```

## 🔧 Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *required* |
| `AGENT_BANANA_IMAGE_MODEL` | Image editing model | `gemini-2.5-flash-preview-04-17` |
| `AGENT_BANANA_ADVISOR_MODEL` | Grounding advisor model | `gemini-2.5-flash-preview-04-17` |

## 📄 Key Concepts from the Paper

### Image Layer Decomposition (ILD)
Instead of editing the full image (which causes color drift and detail loss), ILD:
- Crops the target region with context padding
- Edits only the local patch (model naturally matches surrounding pixels)
- Blends back using Gaussian/Laplacian pyramids

### Context Folding
Compresses interaction history into structured memory across three levels:
- **Asset Level**: Lightweight image state nodes
- **Execution Level**: Transient tool context for error recovery
- **Planning Level**: Persistent memory of verified edit paths

## 📜 License

MIT

## 🙏 Acknowledgments

- [Agent Banana Paper](https://arxiv.org/abs/2602.09084) — Ye et al., 2026
- [Florence-2](https://huggingface.co/microsoft/Florence-2-large) — Microsoft
- [Gemini API](https://ai.google.dev/) — Google DeepMind
