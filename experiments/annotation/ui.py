"""Self-contained local annotation interface.

Annotators receive one HTML file per person.  The page embeds only the blind
items (query + three answers); the answer key with strata and candidate
provenance is never embedded, so opening the file cannot reveal the other
annotator's labels or the sampling design.  Saving uses ``localStorage`` and
export produces JSON/CSV that :mod:`experiments.annotation.store` imports.

No network access, no framework, no build step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from . import schema
from . import store as store_mod

CANDIDATES: tuple[str, ...] = ("c1", "c2", "c3")


def _guidelines_html() -> str:
    parts: list[str] = [
        "<h2 style=\"font-size:16px;margin:0 0 8px\">Annotation guidelines</h2>",
        "<ol>"
        "<li>Work independently. Do not discuss labels with the other annotator before both sheets are submitted.</li>"
        "<li>Label the question the farmer is asking, not the answer.</li>"
        "<li>If two readings are plausible, tick <b>ambiguous</b> and still choose the primary intent.</li>"
        "<li>Use <b>uncertain</b> when you are guessing; certainty is 1 (guess), 2 (likely), 3 (clear).</li>"
        "<li>Judge each candidate answer against the query, not against the other candidates.</li>"
        "<li>Candidate provenance is hidden on purpose: some candidates are unrelated distractors.</li>"
        "<li>Never fill this sheet with an LLM, and never copy another person's labels.</li>"
        "</ol>",
        "<h3>Intent codes</h3><ul>",
    ]
    parts.extend(f"<li><b>{code}</b> — {schema.INTENT_CODES[code]}</li>" for code in schema.INTENT_ORDER)
    parts.append("</ul>")
    parts.append(
        "<h3>Tie-breaks</h3><ul>"
        "<li>Pest/disease control <b>and</b> a dose → PP (the problem comes first).</li>"
        "<li>Pure fertiliser dose with no pest/disease → NM.</li>"
        "<li>A subsidy or loan <b>for</b> a machine or irrigation → GS.</li>"
        "<li>Where to get a variety / which variety → VI; the rate of a seed → MP.</li>"
        "<li>Fodder crop cultivation for cattle → AH; fodder agronomy (sowing time) → CP.</li>"
        "<li>Too short or garbled to tell → OT. Not about agriculture at all (cricket, politics, human health) → OOS.</li>"
        "</ul>"
    )
    for task, labels in (
        ("retrieval_relevance", schema.RELEVANCE_LABELS),
        ("answer_quality", schema.QUALITY_LABELS),
        ("safety", schema.SAFETY_LABELS),
        ("missing_context", schema.CONTEXT_LABELS),
    ):
        parts.append(f"<h3>{task.replace('_', ' ').title()}</h3><ul>")
        parts.extend(
            f"<li><b>{label}</b> — {schema.LABEL_DEFINITIONS[task][label]}</li>" for label in labels
        )
        parts.append("</ul>")
    parts.append(
        "<h3>Submitting</h3>"
        "<p class=\"small\">Export JSON when finished (keep it as your resume file) and export CSV "
        "<code>annotations_&lt;your-name&gt;.csv</code> for import. Export is allowed before every item "
        "is finished; incomplete items are reported by the quality-control script, not silently dropped.</p>"
    )
    return "\n".join(parts)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KCC-Sahayak annotation — __ANNOTATOR__</title>
<style>
:root { --fg:#14210f; --muted:#5b6657; --bg:#f7f8f4; --card:#ffffff; --line:#d8ded1; --accent:#2f6b2a; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.5 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color:var(--fg); background:var(--bg); }
header { position:sticky; top:0; z-index:5; display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:10px 18px; background:var(--card); border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0; }
header .grow { flex:1; }
main { display:grid; grid-template-columns: minmax(0, 1fr) 340px; gap:18px; padding:18px; align-items:start; }
@media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
.panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px 18px; }
.query { font-size:19px; line-height:1.45; padding:14px 16px; border-left:4px solid var(--accent); background:#f1f5ec; border-radius:6px; margin-bottom:14px; white-space:pre-wrap; }
.field { margin:12px 0; }
.label { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin-bottom:6px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { display:inline-flex; align-items:center; gap:6px; border:1px solid var(--line); border-radius:999px; padding:5px 10px; background:#fff; font-size:13px; cursor:pointer; }
.chip input { margin:0; }
select, textarea { font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; background:#fff; color:inherit; max-width:100%; }
textarea { width:100%; }
.row { display:flex; gap:16px; flex-wrap:wrap; align-items:flex-end; }
.cand { border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; background:#fbfcf9; }
.cand h3 { margin:0 0 8px; font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
.cand p { white-space:pre-wrap; margin:0 0 10px; }
button { font:inherit; padding:7px 12px; border-radius:7px; border:1px solid var(--line); background:#fff; cursor:pointer; }
button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.guidelines h3 { font-size:13px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); margin:16px 0 6px; }
.guidelines p, .guidelines li { font-size:13px; }
.small { font-size:12px; color:var(--muted); }
.progress { font-variant-numeric: tabular-nums; font-size:13px; }
.warn { background:#fff6e0; border:1px solid #e6c66a; padding:10px 12px; border-radius:8px; font-size:13px; margin-bottom:14px; }
</style>
</head>
<body>
<header>
  <h1>KCC-Sahayak annotation — __ANNOTATOR__</h1>
  <span class="grow"></span>
  <span class="progress" id="progress"></span>
  <button id="prev">← Previous</button>
  <button id="next">Next →</button>
  <button id="export-json" class="primary">Export JSON</button>
  <button id="export-csv">Export CSV</button>
  <button id="import-btn">Resume from JSON</button>
  <input type="file" id="import" accept=".json,application/json" hidden>
</header>
<main>
  <section>
    <div class="panel">
      <div class="warn">This page contains real KCC query and answer text. Keep the file local and do not paste its contents into a public tool. Do not fill this sheet with an LLM: the study measures human judgement.</div>
      <div class="query" id="query"></div>
      <div id="labels"></div>
      <div id="candidates"></div>
      <div class="small" id="meta"></div>
    </div>
  </section>
  <aside class="panel guidelines">
    __GUIDELINES__
  </aside>
</main>
<script>
const ANNOTATOR = __ANNOTATOR_JSON__;
const ITEMS = __ITEMS_JSON__;
const INTENTS = __INTENTS_JSON__;
const RELEVANCE = __RELEVANCE_JSON__;
const QUALITY = __QUALITY_JSON__;
const SAFETY = __SAFETY_JSON__;
const CONTEXT = __CONTEXT_JSON__;
const CANDIDATES = __CANDIDATES_JSON__;
const STORAGE_KEY = "kcc-annotation:" + ANNOTATOR;

const el = (id) => document.getElementById(id);
let records = loadRecords();
let index = 0;

function loadRecords() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch (error) {
    return {};
  }
}

function saveRecords() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  updateProgress();
}

function recordFor(itemId) {
  if (!records[itemId]) {
    records[itemId] = {
      annotator: ANNOTATOR,
      item_id: itemId,
      intent: "",
      out_of_scope: null,
      multi_intent: [],
      safety: "",
      missing_context: [],
      ambiguous: false,
      uncertain: false,
      certainty: null,
      notes: "",
      retrieval_relevance: {},
      answer_quality: {},
      answer_safety: {}
    };
  }
  const rec = records[itemId];
  rec.multi_intent = rec.multi_intent || [];
  rec.missing_context = rec.missing_context || [];
  rec.retrieval_relevance = rec.retrieval_relevance || {};
  rec.answer_quality = rec.answer_quality || {};
  rec.answer_safety = rec.answer_safety || {};
  return rec;
}

function esc(value) {
  return String(value === null || value === undefined ? "" : value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function optionList(values, current) {
  return values.map((value) => {
    const label = value === "" ? "—" : value;
    return '<option value="' + esc(value) + '"' + (String(current) === String(value) ? " selected" : "") + ">" + esc(label) + "</option>";
  }).join("");
}

function selectHtml(task, candidate, options, rec) {
  const current = (rec[task] || {})[candidate] || "";
  return '<select data-task="' + task + '" data-candidate="' + candidate + '">' +
    optionList([""].concat(options), current) + "</select>";
}

function updateProgress() {
  const annotated = ITEMS.filter((item) => records[item.item_id] && records[item.item_id].intent).length;
  const opened = ITEMS.filter((item) => records[item.item_id]).length;
  el("progress").textContent = annotated + " / " + ITEMS.length + " intents · " + opened + " opened";
}

function render() {
  const item = ITEMS[index];
  const rec = recordFor(item.item_id);
  el("meta").textContent = item.item_id + " · item " + (index + 1) + " of " + ITEMS.length;
  el("query").textContent = item.query;

  let html = '<div class="field"><div class="label">Primary intent</div><div class="chips">' +
    INTENTS.map(([code, name]) =>
      '<label class="chip"><input type="radio" name="intent" value="' + code + '"' +
      (rec.intent === code ? " checked" : "") + "> <b>" + code + "</b> " + esc(name) + "</label>"
    ).join("") + "</div></div>";

  html += '<div class="field"><div class="label">Also asks about (multi-intent; leave empty if single)</div><div class="chips">' +
    INTENTS.map(([code]) =>
      '<label class="chip"><input type="checkbox" data-multi="' + code + '"' +
      ((rec.multi_intent || []).includes(code) ? " checked" : "") + "> " + code + "</label>"
    ).join("") + "</div></div>";

  html += '<div class="row">';
  html += '<div class="field"><div class="label">Out of scope?</div><select name="out_of_scope">' +
    optionList(["", "false", "true"], rec.out_of_scope === true ? "true" : (rec.out_of_scope === false ? "false" : "")) +
    "</select></div>";
  html += '<div class="field"><div class="label">Query safety</div><select name="safety">' +
    optionList([""].concat(SAFETY), rec.safety || "") + "</select></div>";
  html += '<div class="field"><div class="label">Certainty</div><select name="certainty">' +
    optionList(["", "1", "2", "3"], rec.certainty === null ? "" : String(rec.certainty)) + "</select></div>";
  html += "</div>";

  html += '<div class="field"><div class="label">Missing context</div><div class="chips">' +
    CONTEXT.map((value) =>
      '<label class="chip"><input type="checkbox" data-context="' + value + '"' +
      ((rec.missing_context || []).includes(value) ? " checked" : "") + "> " + esc(value) + "</label>"
    ).join("") + "</div></div>";

  html += '<div class="row">' +
    '<label class="chip"><input type="checkbox" name="ambiguous"' + (rec.ambiguous ? " checked" : "") + "> ambiguous</label>" +
    '<label class="chip"><input type="checkbox" name="uncertain"' + (rec.uncertain ? " checked" : "") + "> uncertain</label>" +
    "</div>";

  html += '<div class="field"><div class="label">Notes</div><textarea id="notes" rows="2">' + esc(rec.notes || "") + "</textarea></div>";
  el("labels").innerHTML = html;

  el("candidates").innerHTML = item.candidates.map((candidate) =>
    '<article class="cand"><h3>Candidate ' + esc(candidate.candidate_id.toUpperCase()) + "</h3>" +
    "<p>" + esc(candidate.text) + "</p>" +
    '<div class="row">' +
      '<div class="field"><div class="label">Relevance</div>' + selectHtml("retrieval_relevance", candidate.candidate_id, RELEVANCE, rec) + "</div>" +
      '<div class="field"><div class="label">Answer quality</div>' + selectHtml("answer_quality", candidate.candidate_id, QUALITY, rec) + "</div>" +
      '<div class="field"><div class="label">Answer safety</div>' + selectHtml("answer_safety", candidate.candidate_id, SAFETY, rec) + "</div>" +
    "</div></article>"
  ).join("");

  updateProgress();
}

function toggleMulti(rec, code, checked) {
  rec.multi_intent = (rec.multi_intent || []).filter((value) => value !== code);
  if (checked) rec.multi_intent.push(code);
}

function toggleContext(rec, value, checked) {
  if (value === "none") {
    rec.missing_context = checked ? ["none"] : [];
    return;
  }
  let next = (rec.missing_context || []).filter((item) => item !== "none" && item !== value);
  if (checked) next.push(value);
  rec.missing_context = next;
}

document.addEventListener("change", (event) => {
  const target = event.target;
  const item = ITEMS[index];
  const rec = recordFor(item.item_id);

  if (target.name === "intent") rec.intent = target.value;
  else if (target.name === "out_of_scope") rec.out_of_scope = target.value === "" ? null : target.value === "true";
  else if (target.name === "safety") rec.safety = target.value;
  else if (target.name === "certainty") rec.certainty = target.value ? Number(target.value) : null;
  else if (target.name === "ambiguous") rec.ambiguous = target.checked;
  else if (target.name === "uncertain") rec.uncertain = target.checked;
  else if (target.dataset.multi) toggleMulti(rec, target.dataset.multi, target.checked);
  else if (target.dataset.context) {
    toggleContext(rec, target.dataset.context, target.checked);
    saveRecords();
    render();
    return;
  } else if (target.dataset.task) {
    const task = target.dataset.task;
    rec[task] = rec[task] || {};
    if (target.value) rec[task][target.dataset.candidate] = target.value;
    else delete rec[task][target.dataset.candidate];
  }
  saveRecords();
});

document.addEventListener("input", (event) => {
  if (event.target.id === "notes") {
    const rec = recordFor(ITEMS[index].item_id);
    rec.notes = event.target.value;
    saveRecords();
  }
});

function go(delta) {
  index = Math.max(0, Math.min(ITEMS.length - 1, index + delta));
  render();
}

el("prev").addEventListener("click", () => go(-1));
el("next").addEventListener("click", () => go(1));

document.addEventListener("keydown", (event) => {
  const tag = (event.target.tagName || "").toLowerCase();
  if (["input", "select", "textarea"].includes(tag)) return;
  if (event.key === "ArrowLeft") go(-1);
  if (event.key === "ArrowRight") go(1);
});

function download(name, text, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function exportJSON() {
  const payload = {
    annotator: ANNOTATOR,
    exported_at: new Date().toISOString(),
    protocol: "human annotation, independent, no LLM labels",
    records: Object.values(records).sort((a, b) => a.item_id.localeCompare(b.item_id))
  };
  download("annotations_" + ANNOTATOR + ".json", JSON.stringify(payload, null, 2), "application/json");
}

function quote(value) {
  return '"' + String(value === null || value === undefined ? "" : value).replace(/"/g, '""') + '"';
}

function exportCSV() {
  const header = ["item_id", "intent", "multi_intent", "out_of_scope", "safety", "missing_context",
                  "ambiguous", "uncertain", "certainty", "notes"];
  CANDIDATES.forEach((candidate) => header.push(candidate + "_relevance", candidate + "_quality", candidate + "_safety"));
  const rows = [header.join(",")];
  ITEMS.forEach((item) => {
    const rec = records[item.item_id];
    if (!rec) {
      rows.push([quote(item.item_id)].concat(new Array(header.length - 1).fill('""')).join(","));
      return;
    }
    const row = [
      item.item_id,
      rec.intent || "",
      (rec.multi_intent || []).join(";"),
      rec.out_of_scope === true ? "true" : (rec.out_of_scope === false ? "false" : ""),
      rec.safety || "",
      (rec.missing_context || []).join(";"),
      rec.ambiguous ? "true" : "false",
      rec.uncertain ? "true" : "false",
      rec.certainty === null || rec.certainty === undefined ? "" : rec.certainty,
      rec.notes || ""
    ];
    CANDIDATES.forEach((candidate) => {
      row.push((rec.retrieval_relevance || {})[candidate] || "",
               (rec.answer_quality || {})[candidate] || "",
               (rec.answer_safety || {})[candidate] || "");
    });
    rows.push(row.map(quote).join(","));
  });
  download("annotations_" + ANNOTATOR + ".csv", rows.join("\\n"), "text/csv;charset=utf-8");
}

el("export-json").addEventListener("click", exportJSON);
el("export-csv").addEventListener("click", exportCSV);
el("import-btn").addEventListener("click", () => el("import").click());
el("import").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const payload = JSON.parse(reader.result);
      const incoming = Array.isArray(payload) ? payload : (payload.records || []);
      const known = new Set(ITEMS.map((item) => item.item_id));
      let merged = 0;
      incoming.forEach((rec) => {
        if (!rec || !known.has(rec.item_id)) return;
        records[rec.item_id] = Object.assign({}, records[rec.item_id] || {}, rec, { annotator: ANNOTATOR });
        merged += 1;
      });
      saveRecords();
      render();
      alert("Merged " + merged + " records from " + file.name);
    } catch (error) {
      alert("Could not read this file: " + error.message);
    }
  };
  reader.readAsText(file);
});

render();
</script>
</body>
</html>
"""


def render_ui(annotator: str, items: Sequence[dict]) -> str:
    payload = json.dumps(list(items), ensure_ascii=False).replace("</", "<\\/")
    candidates = sorted({c["candidate_id"] for item in items for c in item.get("candidates", [])})
    return (
        TEMPLATE
        .replace("__ANNOTATOR_JSON__", json.dumps(annotator))
        .replace("__ITEMS_JSON__", payload)
        .replace("__INTENTS_JSON__", json.dumps([[code, name] for code, name in schema.INTENT_CODES.items()], ensure_ascii=False))
        .replace("__RELEVANCE_JSON__", json.dumps(list(schema.RELEVANCE_LABELS)))
        .replace("__QUALITY_JSON__", json.dumps(list(schema.QUALITY_LABELS)))
        .replace("__SAFETY_JSON__", json.dumps(list(schema.SAFETY_LABELS)))
        .replace("__CONTEXT_JSON__", json.dumps(list(schema.CONTEXT_LABELS)))
        .replace("__CANDIDATES_JSON__", json.dumps(candidates))
        .replace("__GUIDELINES__", _guidelines_html())
        .replace("__ANNOTATOR__", annotator)
    )


def write_ui(store: store_mod.Store, annotator: str, overwrite: bool = False) -> Path:
    items = store.load_items()
    if not items:
        raise ValueError("no items found; run build-sample first")
    store.ui_dir.mkdir(parents=True, exist_ok=True)
    path = store.ui_dir / f"{annotator}.html"
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path} (pass --overwrite)")
    path.write_text(render_ui(annotator, items), encoding="utf-8")
    return path
