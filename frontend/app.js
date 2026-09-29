(() => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const recognizeBtn = document.getElementById("recognize-btn");
  const clearBtn = document.getElementById("clear-btn");
  const statusPill = document.getElementById("status-pill");
  const latencyEl = document.getElementById("latency");

  const resultEmpty = document.getElementById("result-empty");
  const resultContent = document.getElementById("result-content");
  const resultError = document.getElementById("result-error");
  const predictedSymbolEl = document.getElementById("predicted-symbol");
  const confidenceBar = document.getElementById("confidence-bar");
  const confidenceValue = document.getElementById("confidence-value");
  const topKEl = document.getElementById("top-k");

  const STROKE_WIDTH = 14;

  let drawing = false;
  let hasInk = false;

  function resetCanvas() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = STROKE_WIDTH;
    hasInk = false;
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

  let lastPoint = null;

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

  function pointerUp(evt) {
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

  function showResult(data) {
    resultEmpty.classList.add("hidden");
    resultError.classList.add("hidden");
    resultContent.classList.remove("hidden");

    renderMath(predictedSymbolEl, data.latex);

    const pct = Math.round(data.confidence * 100);
    confidenceBar.style.width = `${pct}%`;
    confidenceValue.textContent = `${pct}%`;

    topKEl.innerHTML = "";
    data.top_k.forEach((pred) => {
      const row = document.createElement("div");
      row.className = "top-k-row";

      const symbolSpan = document.createElement("span");
      symbolSpan.className = "top-k-symbol";
      renderMath(symbolSpan, pred.latex);

      const barTrack = document.createElement("div");
      barTrack.className = "top-k-bar-track";
      const barFill = document.createElement("div");
      barFill.className = "top-k-bar-fill";
      barFill.style.width = `${Math.round(pred.confidence * 100)}%`;
      barTrack.appendChild(barFill);

      const pctSpan = document.createElement("span");
      pctSpan.style.textAlign = "right";
      pctSpan.textContent = `${Math.round(pred.confidence * 100)}%`;

      row.append(symbolSpan, barTrack, pctSpan);
      topKEl.appendChild(row);
    });
  }

  async function recognize() {
    if (!hasInk) {
      showError("Canvas is empty -- draw a symbol first.");
      return;
    }

    recognizeBtn.disabled = true;
    const start = performance.now();

    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      const formData = new FormData();
      formData.append("file", blob, "symbol.png");

      const response = await fetch("/recognize", { method: "POST", body: formData });
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
  });

  recognizeBtn.addEventListener("click", recognize);

  async function checkHealth() {
    try {
      const response = await fetch("/health");
      const data = await response.json();
      if (data.classifier_loaded) {
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
