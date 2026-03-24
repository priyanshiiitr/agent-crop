from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

from .pipeline import AgentBananaApp
from .vision import decode_image_payload

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Agent Banana Studio</title>
  <style>
    :root {
      --ink: #142126;
      --paper: #f4efe4;
      --panel: rgba(255, 249, 240, 0.88);
      --line: rgba(20, 33, 38, 0.12);
      --olive: #5f6f52;
      --amber: #b45309;
      --copper: #8c3b12;
      --mist: #d8e1cf;
      --shadow: 0 24px 52px rgba(41, 54, 46, 0.12);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Gill Sans", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(95, 111, 82, 0.24), transparent 26%),
        radial-gradient(circle at 85% 10%, rgba(180, 83, 9, 0.16), transparent 22%),
        linear-gradient(140deg, #efe4ce 0%, #f7f2e9 46%, #edf1e6 100%);
      padding: 20px;
    }

    .frame {
      max-width: 1360px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.2fr) minmax(360px, 0.8fr);
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }

    .hero {
      padding: 26px;
      overflow: hidden;
      position: relative;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -60px;
      bottom: -80px;
      width: 220px;
      height: 220px;
      background: radial-gradient(circle, rgba(95, 111, 82, 0.2), transparent 72%);
      pointer-events: none;
    }

    .eyebrow {
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(95, 111, 82, 0.1);
      color: var(--olive);
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-weight: 800;
    }

    h1 {
      margin: 14px 0 10px;
      font-family: "Iowan Old Style", "Book Antiqua", serif;
      font-size: clamp(2.4rem, 5vw, 4.6rem);
      line-height: 0.92;
      max-width: 8ch;
    }

    .lede {
      margin: 0;
      max-width: 60ch;
      color: rgba(20, 33, 38, 0.78);
      line-height: 1.6;
    }

    .composer {
      margin-top: 22px;
      display: grid;
      gap: 14px;
    }

    textarea, input[type="text"] {
      width: 100%;
      border: 1px solid rgba(95, 111, 82, 0.18);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.72);
      padding: 14px 16px;
      color: inherit;
      font: inherit;
      outline: none;
    }

    textarea {
      min-height: 150px;
      resize: vertical;
    }

    textarea:focus, input[type="text"]:focus {
      border-color: rgba(95, 111, 82, 0.5);
      box-shadow: 0 0 0 4px rgba(95, 111, 82, 0.1);
    }

    .toolbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: center;
    }

    .upload {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }

    .button, button {
      border: 0;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--amber) 0%, var(--copper) 100%);
      color: white;
      padding: 12px 18px;
      font: inherit;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 16px 34px rgba(140, 59, 18, 0.18);
    }

    button:disabled { opacity: 0.65; cursor: wait; }

    .sidebar {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
    }

    .card {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.6);
      border-radius: 20px;
      padding: 16px;
    }

    .meta {
      display: grid;
      gap: 6px;
      color: rgba(20, 33, 38, 0.78);
      font-size: 0.96rem;
      line-height: 1.45;
    }

    .gallery {
      grid-column: 1 / -1;
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }

    .image-card {
      padding: 16px;
      display: grid;
      gap: 12px;
    }

    .image-card img {
      width: 100%;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: white;
      min-height: 240px;
      object-fit: contain;
    }

    h2, h3 {
      margin: 0;
      font-size: 1rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--olive);
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      font: inherit;
      line-height: 1.55;
      color: rgba(20, 33, 38, 0.82);
    }

    .plan-list {
      display: grid;
      gap: 10px;
    }

    .plan-item {
      border-radius: 16px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.7);
    }

    .plan-item strong {
      display: block;
      margin-bottom: 4px;
    }

    .hint {
      color: rgba(20, 33, 38, 0.72);
      font-size: 0.95rem;
    }

    @media (max-width: 1020px) {
      .frame {
        grid-template-columns: 1fr;
      }

      body {
        padding: 12px;
      }

      .toolbar {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="frame">
    <section class="panel hero">
      <div class="eyebrow">Planner • Preview • Box • Edit</div>
      <h1>Agent Banana Studio</h1>
      <p class="lede">
        Upload an image, describe the edit, and run the full agent loop: RL-style path search, one Nano Banana preview
        before localization, bounding-box inference, local crop editing, compositing, and quality checks.
      </p>
      <div class="composer">
        <div class="upload">
          <input id="sessionId" type="text" placeholder="Optional session id for multi-turn edits">
          <input id="imageInput" type="file" accept="image/*">
        </div>
        <textarea id="instruction">Replace the center object with a banana and warm up the background lighting.</textarea>
        <div class="toolbar">
          <div class="hint">The app generates one preview image before each bbox decision, then applies the chosen crop edit path.</div>
          <button id="runButton">Run Agent Banana</button>
        </div>
      </div>
    </section>

    <aside class="panel sidebar">
      <div class="card">
        <h2>Status</h2>
        <div id="status" class="meta">Ready.</div>
      </div>
      <div class="card">
        <h2>Folded Context</h2>
        <pre id="contextSummary">No session loaded yet.</pre>
      </div>
      <div class="card">
        <h2>Selected Plan</h2>
        <div id="selectedPlan" class="plan-list"></div>
      </div>
      <div class="card">
        <h2>Candidate Paths</h2>
        <div id="candidatePlans" class="plan-list"></div>
      </div>
    </aside>

    <section class="panel image-card">
      <h2>Preview / Overlay / Final</h2>
      <div class="gallery">
        <div class="card image-card">
          <h3>Source</h3>
          <img id="sourceImage" alt="Source image preview">
        </div>
        <div class="card image-card">
          <h3>Preview Before BBox</h3>
          <img id="previewImage" alt="Preview image">
        </div>
        <div class="card image-card">
          <h3>Detected Region</h3>
          <img id="overlayImage" alt="Bounding box overlay">
        </div>
        <div class="card image-card">
          <h3>Final Output</h3>
          <img id="finalImage" alt="Final edited image">
        </div>
      </div>
    </section>
  </main>

  <script>
    const imageInput = document.getElementById("imageInput");
    const instructionInput = document.getElementById("instruction");
    const sessionInput = document.getElementById("sessionId");
    const runButton = document.getElementById("runButton");
    const statusNode = document.getElementById("status");
    const contextSummary = document.getElementById("contextSummary");
    const selectedPlan = document.getElementById("selectedPlan");
    const candidatePlans = document.getElementById("candidatePlans");
    const sourceImage = document.getElementById("sourceImage");
    const previewImage = document.getElementById("previewImage");
    const overlayImage = document.getElementById("overlayImage");
    const finalImage = document.getElementById("finalImage");

    function dataUrlFromFile(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error("Failed to read the image file."));
        reader.readAsDataURL(file);
      });
    }

    function renderPlanList(node, plans, selectedOnly = false) {
      node.innerHTML = "";
      if (!plans.length) {
        node.textContent = selectedOnly ? "No plan selected." : "No candidate plans yet.";
        return;
      }
      for (const plan of plans) {
        const wrapper = document.createElement("div");
        wrapper.className = "plan-item";
        const stepSummary = plan.steps.map((step) => `${step.order}. ${step.verb} ${step.target} [${step.mode}]`).join("\\n");
        wrapper.innerHTML = `<strong>${plan.plan_id} · score ${plan.score.toFixed(3)}</strong><pre>${stepSummary}</pre>`;
        node.appendChild(wrapper);
      }
    }

    async function runPipeline() {
      const file = imageInput.files[0];
      const instruction = instructionInput.value.trim();
      if (!file) {
        statusNode.textContent = "Choose an image first.";
        return;
      }
      if (!instruction) {
        statusNode.textContent = "Enter an edit instruction.";
        return;
      }

      runButton.disabled = true;
      statusNode.textContent = "Planning, previewing, localizing, and editing...";
      try {
        const imagePayload = await dataUrlFromFile(file);
        sourceImage.src = imagePayload;
        const response = await fetch("/api/edit", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            image: imagePayload,
            instruction,
            session_id: sessionInput.value.trim() || null
          })
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Request failed");
        }

        sessionInput.value = payload.session_id;
        statusNode.textContent = `Completed in image mode ${payload.mode} with grounding ${payload.grounding_mode}. Reward ${payload.reward.toFixed(3)}. Session ${payload.session_id}.`;
        contextSummary.textContent = payload.folded_context.summary;
        renderPlanList(selectedPlan, [payload.selected_plan], true);
        renderPlanList(candidatePlans, payload.candidate_plans.slice(0, 4));

        const firstStep = payload.step_results[0] || null;
        previewImage.src = firstStep ? firstStep.preview_image : payload.source_image;
        overlayImage.src = firstStep ? firstStep.overlay_image : payload.source_image;
        finalImage.src = payload.final_image;
      } catch (error) {
        statusNode.textContent = error.message;
      } finally {
        runButton.disabled = false;
      }
    }

    runButton.addEventListener("click", runPipeline);
  </script>
</body>
</html>
"""


def make_handler(app: AgentBananaApp) -> Callable[..., BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                self._send_html(HTML_PAGE)
                return
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
                return
            self._send_json(404, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/edit":
                self._send_json(404, {"error": "Not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON payload"})
                return

            instruction = str(payload.get("instruction", "")).strip()
            image_payload = str(payload.get("image", "")).strip()
            session_id = payload.get("session_id") or None
            if not instruction:
                self._send_json(400, {"error": "Instruction is required"})
                return
            if not image_payload:
                self._send_json(400, {"error": "Image payload is required"})
                return

            try:
                image = decode_image_payload(image_payload)
                result = app.run(image, instruction, session_id=session_id)
            except Exception as exc:  # pragma: no cover
                self._send_json(500, {"error": str(exc)})
                return

            self._send_json(200, result.to_dict())

        def log_message(self, format: str, *args: object) -> None:
            print(f"[agent-banana] {self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Serve the Agent Banana image editing demo over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    app = AgentBananaApp.from_env(default_root)
    server = HTTPServer((args.host, args.port), make_handler(app))
    print(f"Serving Agent Banana Studio on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
