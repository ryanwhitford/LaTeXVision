(() => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const mainHeat = document.getElementById("heat");

  const recognizeBtn = document.getElementById("recognize-btn");
  const clearBtn = document.getElementById("clear-btn");
  const undoBtn = document.getElementById("undo-btn");
  const statusPill = document.getElementById("status-pill");
  const errorEl = document.getElementById("error");
  const deck = document.getElementById("deck");
  const deckEmpty = document.getElementById("deck-empty");
  const deckCount = document.getElementById("deck-count");
  const newerBtn = document.getElementById("newer-btn");
  const olderBtn = document.getElementById("older-btn");
  const cardTemplate = document.getElementById("card-template");

  const STROKE_WIDTH = 10;
  const STRUCTURAL_TOKENS = new Set(["^", "_", "{", "}", "\\frac"]);
  const STORAGE_KEY = "latexvision.history.v1";
  const MAX_HISTORY = 12;
  const VISIBLE_DEPTH = 3;
  const REPLAY_STEP_MS = 420;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let drawing = false;
  // Every stroke as its list of points, so undo can redraw the rest exactly
  // as they were drawn (same dot + segments, so the model sees the same ink).
  const strokes = [];
  // Bumped on every stroke/undo/clear: a card mirrors its heatmap onto the
  // main canvas only while the canvas still shows the drawing it was made from.
  let canvasVersion = 0;

  const history = []; // oldest first
  let view = -1; // index of the front card
  let replayTimer = null;

  // ---------------------------------------------------------------- drawing

  function paintBackground() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#000000";
    ctx.fillStyle = "#000000";
    ctx.lineWidth = STROKE_WIDTH;
  }

  function drawDot(p) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, STROKE_WIDTH / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawSegment(a, b) {
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  function canvasChanged() {
    canvasVersion++;
    stopReplay();
    clearHeat(mainHeat);
    undoBtn.disabled = strokes.length === 0;
  }

  function resetCanvas() {
    strokes.length = 0;
    paintBackground();
    canvasChanged();
  }

  function undoStroke() {
    if (drawing || strokes.length === 0) return;
    strokes.pop();
    paintBackground();
    for (const points of strokes) {
      drawDot(points[0]);
      for (let i = 1; i < points.length; i++) drawSegment(points[i - 1], points[i]);
    }
    canvasChanged();
  }

  function canvasPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (evt.clientX - rect.left) * (canvas.width / rect.width),
      y: (evt.clientY - rect.top) * (canvas.height / rect.height),
    };
  }

  canvas.addEventListener("pointerdown", (evt) => {
    evt.preventDefault();
    canvas.setPointerCapture(evt.pointerId);
    drawing = true;
    const point = canvasPoint(evt);
    strokes.push([point]);
    drawDot(point);
    canvasChanged();
  });

  canvas.addEventListener("pointermove", (evt) => {
    if (!drawing) return;
    evt.preventDefault();
    const points = strokes[strokes.length - 1];
    const point = canvasPoint(evt);
    drawSegment(points[points.length - 1], point);
    points.push(point);
  });

  for (const type of ["pointerup", "pointercancel", "pointerleave"]) {
    canvas.addEventListener(type, () => {
      drawing = false;
    });
  }

  undoBtn.addEventListener("click", undoStroke);
  document.addEventListener("keydown", (evt) => {
    const target = evt.target;
    if (target instanceof HTMLElement && target.closest("input, textarea, [contenteditable]")) return;
    if ((evt.metaKey || evt.ctrlKey) && !evt.shiftKey && evt.key.toLowerCase() === "z") {
      evt.preventDefault();
      undoStroke();
    }
  });

  function thumbnail() {
    const t = document.createElement("canvas");
    t.width = canvas.width / 2;
    t.height = canvas.height / 2;
    const tctx = t.getContext("2d");
    tctx.drawImage(canvas, 0, 0, t.width, t.height);
    return t.toDataURL("image/png");
  }

  // ---------------------------------------------------------------- heatmaps

  // Deep, saturated stops: the layer multiplies onto white paper, so light
  // tints would vanish. Black ink stays black under multiply.
  const HEAT_STOPS = [
    [0, 196, 255],
    [98, 70, 255],
    [214, 40, 255],
  ];

  function heatColor(t) {
    const x = t * (HEAT_STOPS.length - 1);
    const i = Math.min(Math.floor(x), HEAT_STOPS.length - 2);
    const f = x - i;
    return HEAT_STOPS[i].map((c, k) => Math.round(c + (HEAT_STOPS[i + 1][k] - c) * f));
  }

  function attentionMap(entry, index) {
    entry._maps = entry._maps || [];
    if (!entry._maps[index]) {
      const bin = atob(entry.attention.maps[index]);
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      entry._maps[index] = arr;
    }
    return entry._maps[index];
  }

  function clearHeat(layer) {
    layer.getContext("2d").clearRect(0, 0, layer.width, layer.height);
  }

  // Heat layers share the drawing canvas's 900x260 coordinate space, and the
  // API reports each map's box in that space, so one routine serves both
  // the main canvas and every card thumbnail.
  function drawHeat(layer, entry, index) {
    const lctx = layer.getContext("2d");
    lctx.clearRect(0, 0, layer.width, layer.height);
    if (!entry.attention || index == null) return;
    const { grid_h: gh, grid_w: gw, box } = entry.attention;
    const values = attentionMap(entry, index);

    const small = document.createElement("canvas");
    small.width = gw;
    small.height = gh;
    const sctx = small.getContext("2d");
    const img = sctx.createImageData(gw, gh);
    for (let i = 0; i < values.length; i++) {
      const v = values[i] / 255;
      const t = v < 0.06 ? 0 : Math.pow(v, 0.6); // drop the faint floor, lift the rest
      const [r, g, b] = heatColor(t);
      img.data[i * 4] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = Math.round(t * 255);
    }
    sctx.putImageData(img, 0, 0);

    const [x0, y0, x1, y1] = box;
    const cell = (x1 - x0) / gw;
    lctx.save();
    lctx.imageSmoothingEnabled = true;
    lctx.imageSmoothingQuality = "high";
    lctx.filter = `blur(${Math.max(3, cell * 0.85)}px)`;
    lctx.drawImage(small, x0, y0, x1 - x0, y1 - y0);
    lctx.filter = `blur(${Math.max(2, cell * 0.45)}px)`;
    lctx.globalAlpha = 0.6; // tighter second pass keeps a defined core
    lctx.drawImage(small, x0, y0, x1 - x0, y1 - y0);
    lctx.restore();
  }

  function showToken(entry, card, index) {
    card.querySelectorAll(".token").forEach((el) => el.classList.toggle("active", Number(el.dataset.index) === index));
    drawHeat(card.querySelector(".thumb .heat-layer"), entry, index);
    if (entry.canvasVersion === canvasVersion) drawHeat(mainHeat, entry, index);
  }

  function hideTokens(entry, card) {
    card.querySelectorAll(".token.active").forEach((el) => el.classList.remove("active"));
    clearHeat(card.querySelector(".thumb .heat-layer"));
    if (entry.canvasVersion === canvasVersion) clearHeat(mainHeat);
  }

  function stopReplay() {
    if (replayTimer) {
      clearInterval(replayTimer);
      replayTimer = null;
    }
  }

  function replay(entry, card) {
    stopReplay();
    if (!entry.attention || !entry.tokens.length) return;
    let i = 0;
    showToken(entry, card, i);
    replayTimer = setInterval(() => {
      i++;
      if (i >= entry.tokens.length) {
        stopReplay();
        hideTokens(entry, card);
        return;
      }
      showToken(entry, card, i);
    }, REPLAY_STEP_MS);
  }

  // ---------------------------------------------------------------- cards

  function renderMath(el, latex) {
    // \displaystyle: full-size fractions instead of inline-shrunk ones.
    el.textContent = `$\\displaystyle ${latex}$`;
    const typeset = () => window.MathJax.typesetPromise([el]).catch(() => (el.textContent = latex));
    if (window.MathJax && window.MathJax.typesetPromise) {
      typeset();
    } else {
      // MathJax loads deferred; restored history can render before it's ready.
      let tries = 0;
      const wait = setInterval(() => {
        if ((window.MathJax && window.MathJax.typesetPromise) || ++tries > 50) {
          clearInterval(wait);
          if (window.MathJax && window.MathJax.typesetPromise) typeset();
        }
      }, 100);
    }
  }

  function formatMeta(entry, number) {
    const time = new Date(entry.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    const latency = entry.latencyMs != null ? ` · ${entry.latencyMs} ms` : "";
    return `#${number} · ${time}${latency}`;
  }

  function buildCard(entry, number) {
    const card = cardTemplate.content.firstElementChild.cloneNode(true);
    card.dataset.id = entry.id;
    card.querySelectorAll(".card-meta").forEach((el) => (el.textContent = formatMeta(entry, number)));
    card.querySelector(".thumb img").src = entry.thumb;
    renderMath(card.querySelector(".math"), entry.latex || "\\varnothing");
    card.querySelector(".latex-raw").textContent = entry.latex;
    card.querySelector(".token-count").textContent =
      `${entry.tokens.length} token${entry.tokens.length === 1 ? "" : "s"}: ${entry.tokens.join(" ")}`;

    const list = card.querySelector(".token-list");
    entry.tokens.forEach((token, i) => {
      const chip = document.createElement("span");
      chip.className = STRUCTURAL_TOKENS.has(token) ? "token structural" : "token";
      chip.textContent = token;
      chip.dataset.index = i;
      chip.tabIndex = 0;
      const show = () => {
        stopReplay();
        showToken(entry, card, i);
      };
      chip.addEventListener("pointerenter", show);
      chip.addEventListener("focus", show);
      chip.addEventListener("click", show); // touch: tap to inspect
      list.appendChild(chip);
    });
    list.addEventListener("pointerleave", () => {
      if (!replayTimer) hideTokens(entry, card);
    });
    list.addEventListener("focusout", (evt) => {
      if (!list.contains(evt.relatedTarget) && !replayTimer) hideTokens(entry, card);
    });

    const replayBtn = card.querySelector(".replay-btn");
    if (!entry.attention) {
      replayBtn.classList.add("hidden");
      card.querySelector(".tokens-label").textContent = "Tokens";
    }
    replayBtn.addEventListener("click", () => replay(entry, card));

    card.querySelectorAll(".flip-btn").forEach((btn) =>
      btn.addEventListener("click", () => {
        stopReplay();
        hideTokens(entry, card);
        card.classList.toggle("flipped");
      })
    );

    const status = card.querySelector(".copy-status");
    card.querySelector(".copy-btn").addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(entry.latex);
        status.textContent = "Copied";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(card.querySelector(".latex-raw"));
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        status.textContent = "Selected. Press Cmd/Ctrl+C to copy.";
      }
      setTimeout(() => (status.textContent = ""), 2000);
    });

    // Pointer tilt with a moving specular highlight, front card only.
    card.addEventListener("pointermove", (evt) => {
      if (reducedMotion || evt.pointerType !== "mouse" || card.classList.contains("inactive")) return;
      const r = card.getBoundingClientRect();
      const px = (evt.clientX - r.left) / r.width;
      const py = (evt.clientY - r.top) / r.height;
      card.classList.add("tilting");
      card.style.setProperty("--rx", `${((0.5 - py) * 7).toFixed(2)}deg`);
      card.style.setProperty("--ry", `${((px - 0.5) * 9).toFixed(2)}deg`);
      card.style.setProperty("--mx", `${(px * 100).toFixed(1)}%`);
      card.style.setProperty("--my", `${(py * 100).toFixed(1)}%`);
    });
    card.addEventListener("pointerleave", () => {
      card.classList.remove("tilting");
      card.style.setProperty("--rx", "0deg");
      card.style.setProperty("--ry", "0deg");
    });

    entry._card = card;
    return card;
  }

  // Deck geometry: d = view - index. d = 0 is the front card, d > 0 are
  // older cards stacked up and back, d < 0 are newer cards swept off to the left.
  function layout() {
    deckEmpty.classList.toggle("hidden", history.length > 0);
    history.forEach((entry, index) => {
      const card = entry._card;
      const d = view - index;
      let transform;
      let opacity = 1;
      let filter = "none";
      if (d < 0) {
        transform = "translateX(-115%) translateZ(-120px) rotateY(32deg)";
        opacity = 0;
      } else if (d === 0) {
        transform = "translateZ(0)";
      } else {
        const k = Math.min(d, VISIBLE_DEPTH + 1);
        transform = `translateY(${-k * 15}px) translateZ(${-k * 70}px) rotateX(${k * 2}deg)`;
        opacity = d > VISIBLE_DEPTH ? 0 : 1 - d * 0.2;
        filter = `brightness(${1 - d * 0.16}) saturate(${1 - d * 0.15})`;
      }
      card.style.transform = transform;
      card.style.opacity = opacity;
      card.style.filter = filter;
      card.style.zIndex = String(100 - Math.abs(d));
      card.classList.toggle("inactive", d !== 0);
      card.setAttribute("aria-hidden", d !== 0 ? "true" : "false");
      if (d !== 0) card.classList.remove("flipped");
    });
    const total = history.length;
    deckCount.textContent = total ? `${total - view} / ${total}` : "0 / 0";
    newerBtn.disabled = view >= total - 1;
    olderBtn.disabled = view <= 0;
  }

  function go(delta) {
    const next = Math.max(0, Math.min(history.length - 1, view + delta));
    if (next === view) return;
    stopReplay();
    const leaving = history[view];
    if (leaving) hideTokens(leaving, leaving._card);
    view = next;
    layout();
  }

  newerBtn.addEventListener("click", () => go(1));
  olderBtn.addEventListener("click", () => go(-1));
  deck.addEventListener("keydown", (evt) => {
    if (evt.key === "ArrowRight") go(-1);
    else if (evt.key === "ArrowLeft") go(1);
    else return;
    evt.preventDefault();
  });

  // Swipe left for older results, right for newer.
  let swipeStart = null;
  deck.addEventListener("pointerdown", (evt) => {
    if (evt.target.closest("button, .token, pre")) return;
    swipeStart = { x: evt.clientX, y: evt.clientY };
  });
  deck.addEventListener("pointerup", (evt) => {
    if (!swipeStart) return;
    const dx = evt.clientX - swipeStart.x;
    const dy = evt.clientY - swipeStart.y;
    swipeStart = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5) go(dx < 0 ? -1 : 1);
  });

  function addEntry(entry, { animate }) {
    history.push(entry);
    while (history.length > MAX_HISTORY) {
      const dropped = history.shift();
      dropped._card.remove();
    }
    const card = buildCard(entry, entry.number);
    if (animate && !reducedMotion) {
      // Enter from above the deck, then settle into the front slot.
      card.style.transition = "none";
      card.style.transform = "translateY(-60px) translateZ(120px) rotateX(-12deg)";
      card.style.opacity = "0";
      deck.appendChild(card);
      card.getBoundingClientRect();
      card.style.transition = "";
    } else {
      deck.appendChild(card);
    }
    view = history.length - 1;
    layout();
    return card;
  }

  // ---------------------------------------------------------------- storage

  function save() {
    const plain = (keepAttention) =>
      history.map((e, i) => ({
        id: e.id,
        number: e.number,
        latex: e.latex,
        tokens: e.tokens,
        thumb: e.thumb,
        time: e.time,
        latencyMs: e.latencyMs,
        attention: keepAttention(i) ? e.attention : null,
      }));
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(plain(() => true)));
    } catch {
      try {
        // Over quota: keep heatmaps only for the three newest cards.
        localStorage.setItem(STORAGE_KEY, JSON.stringify(plain((i) => i >= history.length - 3)));
      } catch {
        /* storage unavailable; history lives for this session only */
      }
    }
  }

  function restore() {
    let saved = [];
    try {
      saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    } catch {
      saved = [];
    }
    for (const e of saved.slice(-MAX_HISTORY)) {
      if (!e || !Array.isArray(e.tokens)) continue;
      addEntry({ ...e, canvasVersion: -1 }, { animate: false });
    }
  }

  // ---------------------------------------------------------------- recognize

  function showError(message) {
    errorEl.textContent = message;
    errorEl.classList.remove("hidden");
  }

  async function recognize() {
    errorEl.classList.add("hidden");
    if (strokes.length === 0) {
      showError("The canvas is empty. Write an expression first.");
      return;
    }
    recognizeBtn.disabled = true;
    const start = performance.now();
    const version = canvasVersion;
    const thumb = thumbnail();

    try {
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
      const formData = new FormData();
      formData.append("file", blob, "expression.png");
      const response = await fetch("/recognize-expression?explain=true", { method: "POST", body: formData });
      const latencyMs = Math.round(performance.now() - start);
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        showError(body.detail || `Request failed (${response.status})`);
        return;
      }
      const data = await response.json();
      const last = history[history.length - 1];
      const entry = {
        id: `${Date.now()}`,
        number: (last ? last.number : 0) + 1,
        latex: data.latex,
        tokens: data.tokens,
        attention: data.attention || null,
        thumb,
        time: Date.now(),
        latencyMs,
        canvasVersion: version,
      };
      const card = addEntry(entry, { animate: true });
      save();
      if (!reducedMotion) setTimeout(() => replay(entry, card), 650);
    } catch (err) {
      showError(`Couldn't reach the backend: ${err.message}`);
    } finally {
      recognizeBtn.disabled = false;
    }
  }

  recognizeBtn.addEventListener("click", recognize);
  clearBtn.addEventListener("click", () => {
    stopReplay();
    resetCanvas();
    errorEl.classList.add("hidden");
  });

  async function checkHealth() {
    try {
      const data = await (await fetch("/health")).json();
      statusPill.textContent = data.transformer_loaded ? "model ready" : "no model loaded";
      statusPill.className = `status-pill ${data.transformer_loaded ? "ok" : "error"}`;
    } catch {
      statusPill.textContent = "backend unreachable";
      statusPill.className = "status-pill error";
    }
  }

  resetCanvas();
  restore();
  layout();
  checkHealth();
})();
