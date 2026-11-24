// KCC-Sahayak frontend. Talks to the FastAPI backend under /api and renders
// every figure from the evaluation JSON the backend serves (nothing hard-coded).
(() => {
  "use strict";
  const API = "/api";
  const $ = (id) => document.getElementById(id);
  const state = { stats: null, info: null, samples: [], style: "en", lastId: null,
                  recording: false, recorder: null, chunks: [], selected: null, charts: {} };

  // ---------------------------------------------------------------- helpers
  const pct = (x, d = 1) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
  const num = (x) => (x == null ? "—" : Number(x).toLocaleString());
  const esc = (s) => { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; };
  const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  async function api(path, opts) {
    const r = await fetch(`${API}${path}`, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const REG = { en: "English", hinglish: "Hinglish", hi: "हिंदी", as: "অসমীয়া" };

  // ------------------------------------------------------------------ theme
  const applyTheme = (t) => (t ? document.documentElement.setAttribute("data-theme", t) : document.documentElement.removeAttribute("data-theme"));
  try { applyTheme(localStorage.getItem("kcc_theme")); } catch (_) {}
  $("themeToggle").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const dark = cur === "dark" || (!cur && matchMedia("(prefers-color-scheme: dark)").matches);
    const next = dark ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("kcc_theme", next); } catch (_) {}
    renderCharts();
  });

  let toastTimer;
  function toast(msg) {
    const t = $("toast");
    t.textContent = msg; t.hidden = false;
    requestAnimationFrame(() => t.classList.add("show"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.classList.remove("show"); setTimeout(() => (t.hidden = true), 250); }, 2800);
  }

  // mobile nav
  $("navToggle").addEventListener("click", () => {
    const open = $("navLinks").classList.toggle("open");
    $("navToggle").setAttribute("aria-expanded", String(open));
  });
  document.querySelectorAll("#navLinks a").forEach((a) => a.addEventListener("click", () => $("navLinks").classList.remove("open")));

  // ------------------------------------------------------- hero / about data
  function pick(obj, path) { return path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj); }

  function fillHero() {
    const s = state.stats || {};
    const prod = pick(s, "research_eval.main.production (emb+char, aug)");
    const set = (k, v) => document.querySelectorAll(`[data-stat="${k}"]`).forEach((e) => (e.textContent = v));
    set("corpus", num(state.info && state.info.corpus_size));
    set("silver_acc", prod ? pct(prod.silver.acc) : "—");
    set("gold_acc", prod ? pct(prod.gold_all.acc) : "—");
    const sel = pick(s, "selective_eval.silver_default");
    set("coverage", sel ? pct(sel.coverage) : "—");
    // every figure carries its sample size so it is not read as a bigger claim than it is
    const ci = (o) => (o && o.ci95 ? `, 95% CI ${pct(o.ci95[0])}–${pct(o.ci95[1])}` : "");
    const lab = (k, t) => document.querySelectorAll(`[data-label="${k}"]`).forEach((e) => (e.textContent = t));
    if (prod) {
      lab("silver_acc", `topic accuracy on ${num(prod.silver.n)} held-out calls${ci(prod.silver)}`);
      lab("gold_acc", `topic accuracy on ${num(prod.gold_all.n)} author-written EN / Hinglish / Hindi queries${ci(prod.gold_all)}`);
    }
    if (sel) lab("coverage", `of questions answered fully automatically. The other ${pct(1 - sel.coverage)} get provisional guidance (a safe past answer and an AI suggestion) while a human advisor confirms. Topic correct in ${pct(sel.selective_accuracy)} of the automatic ones`);
  }

  function fillAbout() {
    const s = state.stats || {}, audit = pick(s, "research_eval.corpus_audit") || {}, geo = audit.geography || s.geography || {};
    const ann = s.annotation || {};
    const items = [];
    items.push(`<li><strong>Data.</strong> ${num(audit.corpus_rows || geo.queries)} logged Q/A pairs from a public mirror of the government's KCC corpus. The corpus has no state column; ${num(geo.mention_haryana_places)} queries mention a Haryana place, while about ${num(geo.mention_assam_places_or_terms)} mention Assam places or Assamese crop terms. The content is mostly Assamese, so results are for this multi-state corpus, not one state.</li>`);
    if (ann.n) items.push(`<li><strong>Labels.</strong> ${num(ann.n)} queries were labelled by meaning (LLM-assisted, no human verification yet); the original keyword labels agreed with them only ${pct(ann.keyword_vs_semantic_agreement)} of the time.</li>`);
    if (audit.share_answers_that_are_call_log_notes != null) items.push(`<li><strong>Answers.</strong> ${pct(audit.share_answers_that_are_call_log_notes)} of logged answers are call-log notes (“explained in details”, “transferred to expert”) rather than advice; they are never returned to a farmer.</li>`);
    items.push(`<li><strong>Speech.</strong> Tested only on synthetic English speech so far. Real farmer audio, Hindi and Hinglish speech are untested.</li>`);
    if (state.info) items.push(`<li><strong>Models.</strong> ${esc(state.info.intent_backend)}; retrieval: ${esc(state.info.retrieval_model)} with TF-IDF fusion; ASR: Whisper small.</li>`);
    $("aboutList").innerHTML = items.join("");
  }

  // --------------------------------------------------------------- samples
  function renderChips() {
    const wrap = $("sampleChips");
    wrap.innerHTML = "";
    state.samples.filter((s) => s.style === state.style).forEach((s) => {
      const b = document.createElement("button");
      b.type = "button"; b.className = "chip"; b.textContent = s.label; b.title = s.text;
      b.addEventListener("click", () => { $("textInput").value = s.text; $("textInput").focus(); });
      wrap.appendChild(b);
    });
  }
  document.querySelectorAll(".chip-tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".chip-tab").forEach((x) => x.classList.remove("active"));
    t.classList.add("active"); state.style = t.dataset.style; renderChips();
  }));

  // -------------------------------------------------------- pipeline stages
  const STAGES = ["asr", "intent", "entities", "retrieval", "gate"];
  const stageEl = (s) => document.querySelector(`.stage[data-stage="${s}"]`);
  const outEl = (s) => document.querySelector(`[data-out="${s}"]`);
  function resetStages() {
    STAGES.forEach((s) => stageEl(s).classList.remove("active", "done"));
    outEl("asr").textContent = "Waiting for input…";
    ["intent", "entities", "retrieval", "gate"].forEach((s) => (outEl(s).textContent = "—"));
    $("intentTop3").innerHTML = "";
    document.querySelector('[data-fill="retrieval"]').style.width = "0%";
    $("decisionCard").hidden = true;
  }
  const setStage = (s, st) => { const e = stageEl(s); e.classList.remove("active", "done"); if (st) e.classList.add(st); };

  async function animate(r) {
    resetStages();
    setStage("asr", "active"); await sleep(220);
    outEl("asr").textContent = r.asr ? `“${r.asr.text}” (${r.asr.detected_language})` : `Typed query (${REG[r.register] || r.register}) — speech stage skipped`;
    setStage("asr", "done");

    setStage("intent", "active"); await sleep(260);
    outEl("intent").textContent = `${r.query_type} — ${pct(r.intent_confidence)} confidence`;
    $("intentTop3").innerHTML = r.intent_top3.map((t) =>
      `<div class="t3"><span class="t3-l">${esc(t.label)}</span><span class="t3-bar"><i style="width:${Math.round(t.prob * 100)}%"></i></span><span class="t3-p">${Math.round(t.prob * 100)}%</span></div>`).join("");
    setStage("intent", "done");

    setStage("entities", "active"); await sleep(200);
    const crops = r.entities.crop || [], sym = r.entities.symptom || [], eo = outEl("entities");
    eo.innerHTML = (!crops.length && !sym.length) ? "No crop or pest terms recognised" :
      crops.map((c) => `<span class="entity-tag">${esc(c)}</span>`).join("") + sym.map((c) => `<span class="entity-tag symptom">${esc(c)}</span>`).join("");
    setStage("entities", "done");

    setStage("retrieval", "active"); await sleep(240);
    const top = r.retrieved_top3[0];
    outEl("retrieval").textContent = top && top.matched_question
      ? `${pct(top.similarity, 0)} match${top.source === "advisor" ? " · advisor-verified" : ""} — “${top.matched_question}”`
      : "No usable logged answer found";
    document.querySelector('[data-fill="retrieval"]').style.width = `${Math.max(0, (top ? top.similarity : 0)) * 100}%`;
    setStage("retrieval", "done");

    setStage("gate", "active"); await sleep(220);
    outEl("gate").textContent = r.decision_reason;
    setStage("gate", "done");
    renderDecision(r);
  }

  // The corpus is English. When the farmer asked in Hindi, Hinglish or Assamese, show the answer in that language too.
  // Machine translation (Gemini), always labelled unverified; the English original stays visible.
  function translateBelow(id, text, r) {
    const lang = r.register;
    if (!text || !["hi", "hinglish", "as"].includes(lang)) return;
    const el = $(id), box = document.createElement("div"), qid = r.query_id;
    box.className = "translation"; box.dataset.trFor = id; box.textContent = "Translating…";
    el.before(box);
    post("/translate", { text, lang }).then((t) => {
      if (state.lastId !== qid || !t.text) { box.remove(); return; }
      box.innerHTML = `<small>${esc(REG[lang])} · AI translation, not verified</small><div>${esc(t.text)}</div><small class="orig-label">Original from the KCC corpus (English):</small>`;
      el.classList.add("is-original");
    }).catch(() => box.remove());
  }

  function renderDecision(r) {
    state.lastId = r.query_id;
    document.querySelectorAll(".translation").forEach((e) => e.remove());
    document.querySelectorAll(".is-original").forEach((e) => e.classList.remove("is-original"));
    const card = $("decisionCard");
    card.hidden = false;
    const auto = r.decision === "AUTO_ANSWER";
    card.className = `decision-card ${auto ? "answer" : "escalate"}`;
    $("decisionBadge").textContent = auto ? "✅" : "🚦";
    $("decisionQuestion").textContent = r.translation_error ? `Your question: “${r.input_text}”  ·  translation unavailable (${r.translation_error}), matched directly, so the answer may differ from the English one` : r.original_text ? `Your question: “${r.original_text}”  ·  understood as: “${r.input_text}”` : (r.input_text ? `Your question: “${r.input_text}”` : "");
    $("decisionTitle").textContent = auto ? "Answered automatically"
      : r.decision_code === "time_sensitive" ? "Needs live data: sources below, advisor notified"
      : "Provisional guidance below · an advisor will confirm";
    $("decisionReason").textContent = r.decision_reason;
    $("answerBlock").hidden = !auto;
    $("escalateBlock").hidden = auto;
    if (auto) { $("answerText").textContent = r.suggested_answer; $("fbUp").disabled = $("fbDown").disabled = false; translateBelow("answerText", r.suggested_answer, r); }
    $("candidatesList").innerHTML = r.retrieved_top3.filter((c) => c.matched_question).map((c) =>
      `<li><span class="cand-sim">${pct(c.similarity, 0)}</span> “${esc(c.matched_question)}”${c.source === "advisor" ? ' <span class="badge answer">advisor-verified</span>' : ""}<div class="cand-ans">${esc(c.answer)}</div></li>`).join("");
    loadHistory(); loadQueue(); loadOps();
    card.scrollIntoView({ behavior: "smooth", block: "center" });
    renderCompare(r, auto);
  }

  // Second opinion from Gemini (optional; needs GEMINI_API_KEY on the server). Display only: it never changes the decision.
  function renderCompare(r, auto) {
    const box = $("compareBox"), top = (r.retrieved_top3 || []).find((c) => c.reference_ok);
    box.hidden = false;
    $("cmpCorpus").hidden = auto;     // when auto-answered, the corpus answer is already shown above
    const close = (r.retrieved_top3 || []).find((c) => c.similarity >= 0.55);
    const live = !auto && r.decision_code === "time_sensitive";
    if (live) {
      $("cmpCorpusText").textContent = "Not answerable from the KCC corpus: it is a record of past calls, so it cannot know today's or upcoming weather, and prices change daily and by mandi and region.";
      $("cmpCorpusQ").textContent = "Live data needed: IMD / Meghdoot app for weather, Agmarknet or e-NAM for mandi prices, or a Farm Tele Advisor.";
    } else if (!auto && !top && close) { $("cmpCorpusText").textContent = "The KCC corpus has close past answers, but they name pesticides or doses, so they are withheld here."; $("cmpCorpusQ").textContent = `Closest logged question: “${close.matched_question}” (${pct(close.similarity, 0)} match). A human advisor confirms the product and dose for your case.`; }
    else if (!auto && !top) { $("cmpCorpusText").textContent = "No solid answer in the KCC corpus for this question."; $("cmpCorpusQ").textContent = "The closest past answers were off-topic or were only call-log notes."; }
    if (!auto && top && !live) { translateBelow("cmpCorpusText", top.answer, r); $("cmpCorpusText").textContent = top.answer; $("cmpCorpusQ").textContent = `Matched past question: “${top.matched_question}” (${pct(top.similarity, 0)} match)`; }
    // RAG: Gemini writes the reply from the retrieved KCC answers and cites them. Without safe sources it falls
    // back to plain Gemini advice and the card says so. Display only: never changes the decision.
    const out = $("cmpGeminiText"), id = r.query_id, src = $("cmpSources"), notes = $("cmpNotes"), plainBtn = $("plainGeminiBtn"), plain = $("plainGeminiText");
    out.className = ""; out.textContent = "Reading the KCC answers…";
    src.hidden = true; src.innerHTML = ""; notes.textContent = ""; plainBtn.hidden = true; plain.hidden = true; plain.textContent = "";
    $("cmpGeminiTitle").textContent = "AI answer from KCC sources";
    post("/rag", { text: r.input_text, original_text: r.original_text || r.input_text }).then((g) => {
      if (state.lastId !== id) return;          // a newer query replaced this one
      if (!g.available) { out.className = "cmp-muted"; out.textContent = "Gemini is not set up on this server (add GEMINI_API_KEY to .env)."; return; }
      $("cmpGeminiTitle").textContent = g.grounded ? "AI answer grounded in KCC answers (RAG)" : "Gemini suggestion (general advice, not grounded in the corpus)";
      if (g.withheld) { out.className = "cmp-muted"; out.textContent = "The generated answer contained a dose or quantity, so it is withheld."; }
      else if (g.error || !g.text) { out.className = "cmp-muted"; out.textContent = g.error || "No suggestion returned."; }
      else out.textContent = g.text;
      if (g.grounded && g.sources.length) {
        src.innerHTML = g.sources.map((s) => `<li class="${(g.cited || []).includes(s.n) ? "cited" : ""}">[${s.n}] “${esc(s.question)}” (${pct(s.similarity, 0)}): ${esc(s.answer)}</li>`).join("");
        src.hidden = false;
        plainBtn.hidden = false;
      }
      notes.textContent = (g.notes || []).join(" ");
    }).catch(() => { if (state.lastId === id) { out.className = "cmp-muted"; out.textContent = "Could not get an AI answer."; } });
    plainBtn.onclick = () => {
      plainBtn.disabled = true; plain.hidden = false; plain.textContent = "Asking Gemini without the corpus…";
      post("/gemini", { text: r.original_text || r.input_text }).then((g) => { plain.textContent = "Without the corpus: " + (g.text || g.error || "no answer"); })
        .catch(() => { plain.textContent = "Could not reach Gemini."; }).finally(() => { plainBtn.disabled = false; });
    };
  }

  // ------------------------------------------------------------------ query
  async function run(fn, label) {
    $("submitBtn").disabled = true; $("submitBtnText").textContent = label;
    try { await animate(await fn()); }
    catch (e) { console.error(e); toast(e.message || "Something went wrong"); }
    finally { $("submitBtn").disabled = false; $("submitBtnText").textContent = "Run through pipeline"; }
  }
  $("confSlider").addEventListener("input", () => ($("confVal").textContent = (+$("confSlider").value).toFixed(2)));
  $("simSlider").addEventListener("input", () => ($("simVal").textContent = (+$("simSlider").value).toFixed(2)));
  async function submitText() {
    const text = $("textInput").value.trim();
    if (text.length < 2) return toast("Type or record a question first");
    run(() => post("/query", { text, intent_confidence_threshold: +$("confSlider").value,
      retrieval_similarity_threshold: +$("simSlider").value, screen_answer: $("screenToggle").checked }), "Running…");
  }
  $("submitBtn").addEventListener("click", submitText);
  $("textInput").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submitText(); });
  $("copyAnswerBtn").addEventListener("click", () => navigator.clipboard.writeText($("answerText").textContent).then(() => toast("Answer copied")));
  async function feedback(v) {
    if (!state.lastId) return;
    try { await post(`/queries/${state.lastId}/feedback`, { value: v }); toast("Thanks for the feedback"); $("fbUp").disabled = $("fbDown").disabled = true; loadOps(); }
    catch (e) { toast(e.message); }
  }
  $("fbUp").addEventListener("click", () => feedback(1));
  $("fbDown").addEventListener("click", () => feedback(-1));

  // -------------------------------------------------------------------- mic
  const wave = $("waveCanvas"), wctx = wave.getContext("2d");
  let audioCtx, analyser, raf, stream;
  function idleWave() {
    wctx.clearRect(0, 0, wave.width, wave.height);
    wctx.strokeStyle = cssVar("--border") || "#ccc"; wctx.lineWidth = 2;
    wctx.beginPath(); wctx.moveTo(0, wave.height / 2); wctx.lineTo(wave.width, wave.height / 2); wctx.stroke();
  }
  function liveWave() {
    if (!analyser) return;
    const buf = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteTimeDomainData(buf);
    wctx.clearRect(0, 0, wave.width, wave.height);
    wctx.strokeStyle = "#b3261e"; wctx.lineWidth = 2; wctx.beginPath();
    buf.forEach((v, i) => { const x = (i / buf.length) * wave.width, y = (v / 128) * (wave.height / 2); i ? wctx.lineTo(x, y) : wctx.moveTo(x, y); });
    wctx.stroke(); raf = requestAnimationFrame(liveWave);
  }
  idleWave();
  async function startRec() {
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (_) { return toast("Microphone unavailable or permission denied"); }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioCtx.createAnalyser(); analyser.fftSize = 512;
    audioCtx.createMediaStreamSource(stream).connect(analyser); liveWave();
    state.chunks = []; state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (e) => state.chunks.push(e.data);
    state.recorder.onstop = onRecStop; state.recorder.start();
    state.recording = true; $("micBtn").classList.add("recording"); $("micStatus").textContent = "Listening… tap to stop";
  }
  function stopRec() {
    state.recording = false; $("micBtn").classList.remove("recording");
    if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
    if (stream) stream.getTracks().forEach((t) => t.stop());
    cancelAnimationFrame(raf); idleWave();
  }
  async function onRecStop() {
    $("micStatus").textContent = "Transcribing…";
    const fd = new FormData();
    fd.append("file", new Blob(state.chunks, { type: "audio/webm" }), "query.webm");
    await run(async () => {
      const r = await api("/transcribe", { method: "POST", body: fd });
      $("textInput").value = r.asr ? r.asr.text : r.input_text;
      return r;
    }, "Transcribing…");
    $("micStatus").textContent = "Tap the microphone to speak";
  }
  $("micBtn").addEventListener("click", () => (state.recording ? stopRec() : startRec()));

  // ---------------------------------------------------------------- history
  async function loadHistory() {
    try {
      const { items } = await api("/queries/recent?limit=8");
      $("historyList").innerHTML = items.length ? items.map((h) =>
        `<div class="history-item"><span class="hq">${esc(h.input_text)}</span><span class="badge reg">${esc(REG[h.register] || h.register || "")}</span><span class="badge ${h.decision === "AUTO_ANSWER" ? "answer" : "escalate"}">${h.decision === "AUTO_ANSWER" ? "Answered" : h.status === "resolved" ? "Resolved" : "Escalated"}</span></div>`).join("")
        : '<p class="empty-hint">Queries you run will appear here.</p>';
    } catch (_) {}
  }

  // ---------------------------------------------------------------- advisor
  async function loadQueue() {
    try {
      const [p, r] = await Promise.all([api("/queue?status=pending"), api("/queue?status=resolved&limit=8")]);
      const nb = $("navPending"); nb.textContent = p.items.length; nb.hidden = !p.items.length;
      $("queueList").innerHTML = p.items.length ? p.items.map((q) =>
        `<button class="queue-item${state.selected === q.id ? " sel" : ""}" data-id="${q.id}"><span class="qi-text">${esc(q.input_text)}</span><span class="qi-meta">${esc(q.query_type || "")} · ${esc(REG[q.register] || "")} · ${esc(q.decision_reason || "")}</span></button>`).join("")
        : '<p class="empty-hint">Nothing is waiting. Run a pest or fertiliser question in the demo to see one appear.</p>';
      $("queueList").querySelectorAll(".queue-item").forEach((b) => b.addEventListener("click", () => selectQuery(p.items.find((x) => x.id === +b.dataset.id))));
      $("resolvedList").innerHTML = r.items.length ? r.items.map((q) =>
        `<div class="queue-item done"><span class="qi-text">${esc(q.input_text)}</span><span class="qi-meta">→ ${esc(q.advisor_answer)}</span></div>`).join("") : '<p class="empty-hint">No resolved queries yet.</p>';
      if (state.selected && !p.items.some((x) => x.id === state.selected)) clearSelection();
    } catch (_) {}
  }
  function clearSelection() { state.selected = null; $("resolveForm").hidden = true; $("resolveEmpty").hidden = false; }
  function selectQuery(q) {
    if (!q) return;
    state.selected = q.id; $("resolveEmpty").hidden = true; $("resolveForm").hidden = false;
    $("resolveQ").textContent = q.input_text;
    $("resolveMeta").innerHTML = `<span class="badge reg">${esc(q.query_type || "")}</span><span class="badge reg">${esc(REG[q.register] || "")}</span><span class="badge escalate">${esc(q.decision_code || "escalated")}</span>`;
    $("resolveCandidates").innerHTML = (q.candidates || []).filter((c) => c.matched_question).map((c) =>
      `<div class="resolve-cand"><div class="rc-q">${pct(c.similarity, 0)} · “${esc(c.matched_question)}”</div><div class="rc-a">${esc(c.answer)}</div><button class="btn btn-ghost btn-sm" data-use="${esc(c.answer)}">Use as a starting point</button></div>`).join("") || '<p class="empty-hint">No usable logged answer.</p>';
    $("resolveCandidates").querySelectorAll("[data-use]").forEach((b) => b.addEventListener("click", () => { $("advisorAnswer").value = b.dataset.use; }));
    $("advisorAnswer").value = "";
    loadQueue();
  }
  $("resolveBtn").addEventListener("click", async () => {
    const answer = $("advisorAnswer").value.trim();
    if (answer.length < 3) return toast("Write an answer first");
    try { await post(`/queue/${state.selected}/resolve`, { answer, add_to_memory: $("addMemory").checked }); toast("Answer sent and saved to memory"); clearSelection(); loadQueue(); loadOps(); loadHistory(); }
    catch (e) { toast(e.message); }
  });
  $("refreshQueue").addEventListener("click", loadQueue);

  // -------------------------------------------------------------- operations
  function mk(id, cfg) {
    const el = $(id); if (!el || !window.Chart) return;
    if (state.charts[id]) state.charts[id].destroy();
    state.charts[id] = new Chart(el, cfg);
  }
  const PALETTE = ["#2d6a4f", "#d4a017", "#52b788", "#b3261e", "#a3730a", "#3a8462", "#f0cd6b", "#7a1a15", "#948c76", "#6b6350"];
  const baseOpts = () => ({ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
    scales: { x: { ticks: { color: cssVar("--text-muted") }, grid: { display: false } }, y: { ticks: { color: cssVar("--text-muted") }, grid: { color: cssVar("--border") } } } });
  let ops = null;
  async function loadOps() { try { ops = await api("/ops/summary"); renderOps(); } catch (_) {} }
  function renderOps() {
    if (!ops) return;
    const set = (k, v) => document.querySelectorAll(`[data-ops="${k}"]`).forEach((e) => (e.textContent = v));
    set("total", num(ops.total)); set("automation_rate", pct(ops.automation_rate)); set("pending", num(ops.pending));
    set("resolved", num(ops.resolved)); set("avg_latency_ms", ops.avg_latency_ms == null ? "—" : `${Math.round(ops.avg_latency_ms)} ms`);
    const fb = ops.feedback_up + ops.feedback_down; set("satisfaction", fb ? pct(ops.feedback_up / fb, 0) : "—");
    const t = Object.entries(ops.by_type || {});
    mk("opsTypeChart", { type: "bar", data: { labels: t.map((x) => x[0] || "—"), datasets: [{ data: t.map((x) => x[1]), backgroundColor: PALETTE, borderRadius: 6 }] }, options: { ...baseOpts(), indexAxis: "y" } });
    const r = Object.entries(ops.by_register || {});
    mk("opsRegChart", { type: "doughnut", data: { labels: r.map((x) => REG[x[0]] || x[0]), datasets: [{ data: r.map((x) => x[1]), backgroundColor: PALETTE }] },
      options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom", labels: { color: cssVar("--text-muted") } } } } });
  }

  // -------------------------------------------------------------- evaluation
  function renderEval() {
    const s = state.stats || {}, re = s.research_eval, se = s.selective_eval;
    if (!re) { $("evalCallouts").innerHTML = '<p class="empty-hint">Evaluation results are not available yet (run scripts/17_research_eval.py).</p>'; return; }
    const ag = re.agreement_vs_accuracy || {}, au = re.corpus_audit || {}, geo = au.geography || {}, ann = s.annotation || {};
    const cards = [
      [pct(ann.keyword_vs_semantic_agreement), "of the original keyword labels agreed with labels assigned by meaning"],
      [`${pct(ag.old_distilbert_agreement_with_keyword_labels)} → ${pct(ag.old_distilbert_accuracy_vs_meaning_labels_silver)}`, "the original model's score against keyword labels, then against meaning-based labels"],
      [pct(au["weak_test_near_duplicate_share_ge_0.95"], 0), `of old test queries had a near-duplicate in training; retrieval matched ${pct(au.retrieval_category_match_near_duplicate, 0)} on those but ${pct(au.retrieval_category_match_novel, 0)} on novel ones`],
      [`${num(geo.mention_haryana_places)} / ${num(geo.queries)}`, "queries mention a Haryana place; the corpus is overwhelmingly Assamese"],
      [pct(au.share_answers_that_are_call_log_notes), "of logged answers are call-log notes (“explained in details”), not advice"],
    ];
    $("evalCallouts").innerHTML = cards.map((c) => `<div class="callout-card"><div class="cc-num">${c[0]}</div><div class="cc-txt">${c[1]}</div></div>`).join("");

    const rows = [["keyword rules (original)", "Keyword rules (original labeller)"], ["DistilBERT on keyword labels (original)", "DistilBERT on keyword labels (original model)"],
      ["word TF-IDF + SVM", "Word TF-IDF + SVM"], ["char TF-IDF + SVM", "Character TF-IDF + SVM"], ["embedding + LR", "Sentence embedding + LR"], ["production (emb+char, aug)", "Embedding + character TF-IDF + LR (deployed)"]];
    const cell = (o) => (o ? `<td>${pct(o.acc, 1)}<span class="ci">${o.ci95 ? `${(o.ci95[0] * 100).toFixed(0)}–${(o.ci95[1] * 100).toFixed(0)}` : ""}</span></td>` : "<td>—</td>");
    $("intentTable").innerHTML = `<thead><tr><th>System</th><th>Silver</th><th>Gold (all)</th><th>English</th><th>Hinglish</th><th>Hindi</th><th>Hinglish*</th></tr></thead><tbody>` +
      rows.map(([k, label]) => { const r = re.main[k]; if (!r) return ""; const hp = r.hinglish_perturbed; return `<tr class="${k.startsWith("production") ? "hl" : ""}"><td>${label}</td>${cell(r.silver)}${cell(r.gold_all)}${cell(r.gold_en)}${cell(r.gold_hinglish)}${cell(r.gold_hi)}<td>${hp ? pct(hp.acc, 1) : "—"}</td></tr>`; }).join("") + "</tbody>";

    if (se && se.silver_sweep) {
      const sw = se.silver_sweep.filter((x) => x.selective_accuracy != null && Math.abs(x.sim - 0.5) < 0.11);
      const pts = sw.map((x) => ({ x: x.coverage * 100, y: x.selective_accuracy * 100 }));
      mk("selChart", { type: "scatter", data: { datasets: [{ label: "gate settings", data: pts, backgroundColor: PALETTE[0] }] },
        options: { ...baseOpts(), scales: { x: { title: { display: true, text: "Auto-answered (%)", color: cssVar("--text-muted") }, ticks: { color: cssVar("--text-muted") } },
          y: { title: { display: true, text: "Accuracy of auto-answered subset (%)", color: cssVar("--text-muted") }, ticks: { color: cssVar("--text-muted") }, grid: { color: cssVar("--border") } } } } });
      const d = se.silver_default; $("selNote").textContent = d ? `At the default gate, ${pct(d.coverage, 0)} of held-out queries are answered automatically with ${pct(d.selective_accuracy)} intent accuracy (n=${d.n}).` : "";
    }
    mk("agreeChart", { type: "bar", data: { labels: ["Original model vs keyword labels", "Original model vs meaning labels", "Deployed model vs meaning labels"],
      datasets: [{ data: [ag.old_distilbert_agreement_with_keyword_labels * 100, ag.old_distilbert_accuracy_vs_meaning_labels_silver * 100, ag.production_accuracy_vs_meaning_labels_silver * 100], backgroundColor: [PALETTE[3], PALETTE[1], PALETTE[0]], borderRadius: 6 }] },
      options: { ...baseOpts(), scales: { y: { min: 0, max: 100, ticks: { color: cssVar("--text-muted") }, grid: { color: cssVar("--border") } }, x: { ticks: { color: cssVar("--text-muted") }, grid: { display: false } } } } });
    $("agreeNote").textContent = "The original 99% was agreement with keyword labels, which themselves disagree with meaning a third of the time.";

    const facts = [];
    const ab = se && se.gate_ablation_silver;
    if (ab) { const f = ab["5_+topic_match(full)"], t = ab["1_thresholds_only"]; facts.push(["Safety gate ablation (held-out calls)", `Thresholds alone auto-answer ${pct(t.coverage, 0)} of queries, but ${pct(t.violation_rate_among_auto, 0)} of those break policy (a high-stakes or time-sensitive topic, or risky answer text). The full gate answers ${pct(f.coverage, 0)} with ${pct(f.violation_rate_among_auto, 0)} violations.`]); }
    const cal = s.calibration; if (cal) facts.push(["Calibration", `Temperature scaling (T=${cal.temperature.toFixed(2)}) cut out-of-fold expected calibration error from ${pct(cal.ece_oof_uncalibrated, 1)} to ${pct(cal.ece_oof_calibrated, 1)}. On the held-out set it was already well calibrated (${pct(cal.ece_silver_uncalibrated, 1)} → ${pct(cal.ece_silver_calibrated, 1)}).`]);
    const hy = s.hybrid_retrieval; if (hy) { const a = hy.alphas[String(hy.chosen_alpha)], d = hy.alphas["1.0"]; if (a && d) facts.push(["Hybrid retrieval", `Fusing TF-IDF with dense search (weight ${hy.chosen_alpha}) raised answer F1 against the advisor's real answer from ${d.all.toFixed(3)} to ${a.all.toFixed(3)} overall and ${d.novel.toFixed(3)} to ${a.novel.toFixed(3)} on novel queries (n=${hy.n}).`]); }
    const v = s.voice_eval; if (v) facts.push(["Voice", `Whisper small on synthetic English speech: word error rate ${pct(v.mean_wer)}; intent accuracy ${pct(v.intent_accuracy_typed_text, 0)} on typed text vs ${pct(v.intent_accuracy_after_asr, 0)} after speech recognition (n=${v.n}). Real phone-line speech will be harder.`]);
    const mc = re.mcnemar_vs_production && re.mcnemar_vs_production["char TF-IDF + SVM"];
    if (mc) facts.push(["Is the deployed model really better?", `Not significantly better than a plain character n-gram SVM (McNemar p = ${mc.silver.p_value.toFixed(2)} held-out, ${mc.gold_all.p_value.toFixed(2)} gold). It is clearly better than word TF-IDF and than embeddings alone. The embedding's real value is transfer to Hindi when no Hindi training text exists.`]);
    const ec = s.encoder_comparison;
    if (ec && ec.results) {
      const rows = Object.entries(ec.results).filter(([k]) => k.includes("char-tfidf") || k.includes("classification head"))
        .map(([k, x]) => `${k.split(" |")[0]}: ${pct(x.silver, 1)} held-out, ${pct(x.gold, 1)} gold`);
      facts.push(["Encoder comparison (MuRIL / IndicBERT vs the deployed MiniLM)", `${rows.join(" · ")}. Same training data and head for every encoder${ec.partial ? " (partial run)" : ""}. The gold set has only n=${ec.n_gold}, so gaps of a few points are within noise.`]);
    }
    const rb = s.retrieval_baselines;
    if (rb && rb.systems) facts.push(["Retrieval vs a KisanQRS-style baseline", Object.entries(rb.systems).map(([k, x]) => `${k}: F1 ${x.f1.toFixed(3)}, ${pct(x.call_log_note_rate, 1)} call-log notes returned`).join(" · ") + `. The KisanQRS-style row is a simplified re-implementation of its retrieval idea (n=${rb.n}).`]);
    const nv = re.novelty_silver; if (nv) facts.push(["Novel vs repeated questions", Object.entries(nv).map(([k, x]) => `${k}: ${pct(x.accuracy, 0)} (n=${x.n})`).join(" · ")]);
    const lc = re.learning_curve; if (lc) { const ks = Object.keys(lc); facts.push(["How much annotation is needed", `Held-out accuracy rises from ${pct(lc[ks[0]].silver_mean, 0)} with ${ks[0]} labelled queries to ${pct(lc[ks[ks.length - 1]].silver_mean, 0)} with ${ks[ks.length - 1]}.`]); }
    $("evalFacts").innerHTML = facts.map((f) => `<div class="fact"><h4>${f[0]}</h4><p>${f[1]}</p></div>`).join("");
  }

  function renderCharts() { renderOps(); renderEval(); }

  // -------------------------------------------------------------------- init
  async function init() {
    renderChips();
    const [stats, info, samples] = await Promise.allSettled([api("/stats"), api("/model-info"), api("/samples")]);
    state.stats = stats.value || {}; state.info = info.value || null; state.samples = (samples.value && samples.value.samples) || [];
    $("intentTag").textContent = state.info ? state.info.intent_backend : "classifier";
    renderChips(); fillHero(); fillAbout(); renderEval();
    loadHistory(); loadQueue(); loadOps();
    setInterval(loadQueue, 20000);
  }
  init();
})();
