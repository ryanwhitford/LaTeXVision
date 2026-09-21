(() => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const overlay = document.getElementById("overlay");
  const overlayCtx = overlay.getContext("2d");

  const recognizeBtn = document.getElementById("recognize-btn");
  const clearBtn = document.getElementById("clear-btn");
  const overlayToggle = document.getElementById("overlay-toggle");
  const statusPill = document.getElementById("status-pill");
  const latencyEl = document.getElementById("latency");

  const resultEmpty = document.getElementById("result-empty");
  const resultContent = document.getElementById("result-content");
  const resultError = document.getElementById("result-error");
  const predictedLatexEl = document.getElementById("predicted-latex");
  const latexRawEl = document.getElementById("latex-raw");
  const symbolListEl = document.getElementById("symbol-list");

  const STROKE_WIDTH = 10;
  const ROLE_COLORS = {
    base: "#4fd1c5",
    superscript: "#f6ad55",
    subscript: "#b794f4",
    operator: "#8891a7",
  };

  let drawing = false;
  let hasInk = false;
  let lastPoint = null;
  let lastSymbols = [];

  function resetCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = STROKE_WIDTH;
    hasInk = false;
    clearOverlay();
  }

  function clearOverlay() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function canvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (evt.clientX - rect.left) * scaleX,
      y: (evt.clientY - rect.top) * scaleY,
    };
  }

  function pointerDown(evt) {
    evt.preventDefault();
    canvas.setPointerCapture(evt.pointerId);
    drawing = true;
    hasInk = true;
    lastPoint = canvasPoint(evt);
    ctx.beginPath();
    ctx.arc(lastPoint.x, lastPoint.y, STROKE_WIDTH / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#000000";
    ctx.fill();
  }

  function pointerMove(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const point = canvasPoint(evt);
    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint = point;
  }

  function pointerUp() {
    drawing = false;
    lastPoint = null;
  }

  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", pointerUp);
  canvas.addEventListener("pointerleave", pointerUp);

  function showEmpty() {
    resultEmpty.classList.remove("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.add("hidden");
  }

  function showError(message) {
    resultEmpty.classList.add("hidden");
    resultContent.classList.add("hidden");
    resultError.classList.remove("hidden");
    resultError.textContent = message;
  }

  function renderMath(el, latex) {
    el.textContent = `$${latex}$`;
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([el]).catch(() => {
        el.textContent = latex;
      });
    }
  }

  function drawOverlay(symbols) {
    clearOverlay();
    if (!overlayToggle.checked) return;
    const scaleX = overlay.width / canvas.width;
    const scaleY = overlay.height / canvas.height;
    symbols.forEach((s) => {
      const [x1, y1, x2, y2] = s.bbox;
      const color = ROLE_COLORS[s.role] || "#8891a7";
      overlayCtx.strokeStyle = color;
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
      overlayCtx.fillStyle = color;
      overlayCtx.font = "11px monospace";
      overlayCtx.fillText(s.role, x1 * scaleX + 2, y1 * scaleY - 4 < 10 ? y2 * scaleY + 13 : y1 * scaleY - 4);
    });
  }

  function showResult(data) {
    resultEmpty.classList.add("hidden");
    resultError.classList.add("hidden");
    resultContent.classList.remove("hidden");

    renderMath(predictedLatexEl, data.latex);
    latexRawEl.textContent = data.latex;

    symbolListEl.innerHTML = "";
    data.symbols.forEach((s) => {
      const chip = document.createElement("div");
      chip.className = `symbol-chip role-${s.role}`;

      const symbolSpan = document.createElement("span");
      symbolSpan.className = "chip-symbol";
      renderMath(symbolSpan, s.latex);

      const roleSpan = document.createElement("span");
      roleSpan.className = "chip-role";
      roleSpan.textContent = s.attached_to ? `${s.role} of ${s.attached_to}` : s.role;

      const confSpan = document.createElement("span");
      confSpan.textContent = `${Math.round(s.confidence * 100)}%`;

      chip.append(symbolSpan, roleSpan, confSpan);
      symbolListEl.appendChild(chip);
    });

    lastSymbols = data.symbols;
    drawOverlay(lastSymbols);
  }

  async function recognize() {
    if (!hasInk) {
      showError("Canvas is empty -- write an expression first.");
      return;
    }

    recognizeBtn.disabled = true;
    const start = performance.now();

    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      const formData = new FormData();
      formData.append("file", blob, "expression.png");

      const response = await fetch("/recognize-expression", { method: "POST", body: formData });
      const elapsed = Math.round(performance.now() - start);

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        showError(errBody.detail || `Request failed (${response.status})`);
        latencyEl.textContent = "";
        return;
      }

      const data = await response.json();
      showResult(data);
      latencyEl.textContent = `${elapsed}ms`;
    } catch (err) {
      showError(`Could not reach the backend: ${err.message}`);
    } finally {
      recognizeBtn.disabled = false;
    }
  }

  clearBtn.addEventListener("click", () => {
    resetCanvas();
    showEmpty();
    latencyEl.textContent = "";
    lastSymbols = [];
  });

  overlayToggle.addEventListener("change", () => drawOverlay(lastSymbols));

  recognizeBtn.addEventListener("click", recognize);

  async function checkHealth() {
    try {
      const response = await fetch("/health");
      const data = await response.json();
      if (data.model_loaded) {
        statusPill.textContent = "model ready";
        statusPill.className = "status-pill ok";
      } else {
        statusPill.textContent = "no model loaded";
        statusPill.className = "status-pill error";
      }
    } catch {
      statusPill.textContent = "backend unreachable";
      statusPill.className = "status-pill error";
    }
  }

  resetCanvas();
  checkHealth();
})();
