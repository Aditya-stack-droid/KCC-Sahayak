"""Generate all diagrams and result figures for the KCC-Sahayak report.

Run from the project root:
    .venv312/bin/python KCC_Sahayak_Diagrams/make_diagrams.py

Outputs PNG files (200 dpi) into KCC_Sahayak_Diagrams/.
Mermaid source files (.mmd) are written alongside for editable versions.
"""
from pathlib import Path

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "KCC_Sahayak_Diagrams"
OUT.mkdir(exist_ok=True)

# palette
INK = "#1f2937"
MUTED = "#6b7280"
BLUE = "#2563eb"
BLUE_L = "#dbeafe"
GREEN = "#059669"
GREEN_L = "#d1fae5"
AMBER = "#b45309"
AMBER_L = "#fef3c7"
RED = "#b91c1c"
RED_L = "#fee2e2"
GREY_L = "#f3f4f6"
PURPLE = "#6d28d9"
PURPLE_L = "#ede9fe"
TEAL = "#0f766e"
TEAL_L = "#ccfbf1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#d1d5db",
    "axes.labelcolor": INK,
    "text.color": INK,
})


def box(ax, x, y, w, h, text, fc=GREY_L, ec=INK, fs=9, bold=False, tc=INK, rounded=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={rounded}",
                       linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)


def arrow(ax, p1, p2, color=INK, style="-|>", lw=1.3, rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, linewidth=lw,
                        color=color, connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)


def canvas(w=13, h=8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. architecture
def architecture():
    fig, ax = canvas(14, 9.5)
    ax.text(50, 97, "KCC-Sahayak — Layered System Architecture", ha="center", fontsize=15,
            fontweight="bold")

    # presentation layer
    box(ax, 2, 78, 96, 14, "", fc="#f9fafb", ec="#d1d5db")
    ax.text(4, 89.5, "PRESENTATION LAYER (single origin, static HTML/CSS/JS + Chart.js)", fontsize=8.5,
            color=MUTED, fontweight="bold")
    box(ax, 4, 80, 21, 7.5, "Farmer Live Demo\nvoice upload / text query", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 27, 80, 21, 7.5, "Advisor Console\nescalation queue + answers", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 50, 80, 21, 7.5, "Operations Dashboard\nlive logs, coverage, latency", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 73, 80, 23, 7.5, "Evaluation Dashboard\nreads results/*.json", fc=BLUE_L, ec=BLUE, bold=True)

    # api layer
    box(ax, 2, 61, 96, 13, "", fc="#f9fafb", ec="#d1d5db")
    ax.text(4, 71.5, "API LAYER — FastAPI (backend/app.py), JSON over HTTP", fontsize=8.5, color=MUTED,
            fontweight="bold")
    api = [("POST /api/query", 4, 16), ("POST /api/transcribe", 22, 17),
           ("GET /api/queue\nPOST /queue/{id}/resolve", 41, 21),
           ("POST /queries/{id}/feedback", 64, 18),
           ("GET /api/ops/summary\nGET /api/stats", 84, 12)]
    for t, x, w in api:
        box(ax, x, 63, w, 6.5, t, fc=GREEN_L, ec=GREEN, fs=7.2)

    # pipeline layer
    box(ax, 2, 32, 96, 26, "", fc="#f9fafb", ec="#d1d5db")
    ax.text(4, 56.7, "NLP & DECISION LAYER — src/pipeline.py (one process, models loaded once)",
            fontsize=8.5, color=MUTED, fontweight="bold")
    box(ax, 4, 46, 16, 8.5, "Language / register\ndetection\n(lang.py)", fc=PURPLE_L, ec=PURPLE)
    box(ax, 22, 46, 16, 8.5, "Whisper ASR\n(voice only)\n(asr.py)", fc=PURPLE_L, ec=PURPLE)
    box(ax, 40, 46, 17, 8.5, "Multilingual encoder\nMiniLM-L12-v2\n(384-d, L2 norm)", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 59, 46, 18, 8.5, "Intent head\nembedding + char 2–5 TF-IDF\n→ logistic regression\n+ temperature scaling", fc=GREEN_L, ec=GREEN, bold=True, fs=7.6)
    box(ax, 79, 46, 17, 8.5, "Entity tagger\ncrop / pest gazetteer\n(ner.py)", fc=PURPLE_L, ec=PURPLE)
    box(ax, 4, 34, 27, 8.5, "Hybrid retriever\nFAISS dense top-200 ∪ TF-IDF top-200\nfused = α·cos + (1−α)·tfidf", fc=AMBER_L, ec=AMBER, bold=True)
    box(ax, 33, 34, 27, 8.5, "Safety gate (confidence_gate.py)\nhigh-stakes · time-sensitive · confidence ·\nsimilarity · topic match · answer screening", fc=RED_L, ec=RED, bold=True, fs=7.6)
    box(ax, 62, 34, 15, 8.5, "AUTO_ANSWER\nretrieved advisor text\n+ feedback", fc=GREEN_L, ec=GREEN, bold=True)
    box(ax, 79, 34, 17, 8.5, "ESCALATE\nreason code + advisor\nqueue (human FTA)", fc=RED_L, ec=RED, bold=True)

    # data layer
    box(ax, 2, 5, 96, 24, "", fc="#f9fafb", ec="#d1d5db")
    ax.text(4, 27.7, "DATA & ARTEFACT LAYER", fontsize=8.5, color=MUTED, fontweight="bold")
    box(ax, 4, 16, 20, 9.5, "Raw KCC corpus\ndata/raw/\nkcc_mapping_full.csv\n139,542 Q/A pairs", fc=GREY_L)
    box(ax, 26, 16, 20, 9.5, "Processed splits\ntrain 111,100 · val 13,887\ntest 13,888 (80/10/10)", fc=GREY_L)
    box(ax, 48, 16, 22, 9.5, "Indexes & models\nFAISS 124,987 vectors\ntfidf matrix · LR head", fc=GREY_L)
    box(ax, 72, 16, 24, 9.5, "SQLite app.db\nquery audit log + advisor\nqueue + feedback", fc=GREY_L)
    box(ax, 4, 7, 28, 6.5, "results/*.json  (all reported metrics)", fc="#ffffff", ec="#9ca3af", fs=8)
    box(ax, 34, 7, 30, 6.5, "data/gold/gold_queries.csv (150)", fc="#ffffff", ec="#9ca3af", fs=8)
    box(ax, 66, 7, 30, 6.5, "data/annotation/ (3,496 labels + 1,200 paraphrases)", fc="#ffffff", ec="#9ca3af", fs=8)

    # cross-layer arrows
    for x in (14, 37, 60, 84):
        arrow(ax, (x, 80), (x, 74.5), color=MUTED)
    arrow(ax, (50, 61), (50, 58.5), color=MUTED)
    arrow(ax, (48.5, 46), (48.5, 43), color=BLUE)
    arrow(ax, (50, 46), (44, 43), color=BLUE, rad=-0.1)
    arrow(ax, (77.5, 46), (77.5, 43), color=GREEN)
    arrow(ax, (31.2, 38), (32.8, 38), color=AMBER)
    arrow(ax, (60.2, 38), (61.8, 38), color=RED)
    arrow(ax, (76.5, 38), (78.8, 38), color=GREEN)
    arrow(ax, (50, 32), (50, 29.8), color=MUTED)
    arrow(ax, (90, 80), (90, 74.5), color=MUTED)
    save(fig, "01_system_architecture.png")


# ---------------------------------------------------------------- 2. workflow
def workflow():
    fig, ax = canvas(11, 14)
    ax.text(50, 98, "End-to-End Query Workflow", ha="center", fontsize=15, fontweight="bold")
    steps = [
        (90, "Farmer input\nvoice file or typed text", BLUE_L, BLUE),
        (81, "Register detection (hi / hinglish / en)\noptional Whisper transcription + normalisation", PURPLE_L, PURPLE),
        (72, "Sentence embedding (MiniLM, 384-d, normalised)", BLUE_L, BLUE),
        (63, "Intent classification (calibrated LR)\n+ crop / pest entity tagging", GREEN_L, GREEN),
    ]
    for y, t, fc, ec in steps:
        box(ax, 22, y, 56, 6.5, t, fc=fc, ec=ec)
    box(ax, 22, 53, 56, 6.5, "Hybrid retrieval over 124,987 logged Q/A pairs\n(FAISS dense ∪ TF-IDF; informative answers only; advisor memory)",
        fc=AMBER_L, ec=AMBER)
    box(ax, 22, 43.5, 56, 7.5, "SAFETY GATE (all checks must pass)\n1 not high-stakes  2 not time-sensitive  3 confidence ≥ 0.70\n4 similarity ≥ 0.55  5 same topic  6 real advice  7 no dose/chemical",
        fc=RED_L, ec=RED, bold=True)
    box(ax, 4, 30, 40, 8.5, "AUTO_ANSWER\nreturn closest logged advisor answer\n(logged in SQLite, feedback ±1)", fc=GREEN_L, ec=GREEN, bold=True)
    box(ax, 56, 30, 40, 8.5, "ESCALATE\nreason code shown to farmer; query enters\npending human-advisor queue", fc=RED_L, ec=RED, bold=True)
    box(ax, 56, 17, 40, 8, "Advisor Console\nhuman FTA writes an answer", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 22, 5, 56, 7.5, "Advisor answer returned to farmer AND appended to\nin-process retrieval memory (human-in-the-loop)",
        fc=TEAL_L, ec=TEAL, bold=True)

    ys = [90, 81, 72, 63, 53]
    for i in range(len(ys) - 1):
        arrow(ax, (50, ys[i]), (50, ys[i + 1] + 6.5))
    arrow(ax, (50, 53), (50, 51.2))
    arrow(ax, (40, 43.5), (26, 38.6), color=GREEN)
    arrow(ax, (60, 43.5), (74, 38.6), color=RED)
    arrow(ax, (76, 30), (76, 25.1), color=RED)
    arrow(ax, (70, 17), (64, 12.6), color=TEAL)
    ax.text(50, 1.5, "The same answer is appended to advisor memory and is searched in future retrievals.",
            fontsize=7.8, color=TEAL, ha="center")
    save(fig, "02_query_workflow.png")


# ---------------------------------------------------------------- 3. model architecture
def model_arch():
    fig, ax = canvas(12.5, 7.2)
    ax.text(50, 95, "Intent Classification Architecture (deployed model)", ha="center", fontsize=14,
            fontweight="bold")
    box(ax, 3, 62, 20, 14, "Input query\n(English /\nHinglish / Hindi)", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 28, 62, 20, 14, "Frozen MiniLM\nmultilingual\nencoder → 384-d\nL2-normalised", fc=PURPLE_L, ec=PURPLE, bold=True)
    box(ax, 55, 76, 20, 10, "Dense block\n384 features × w", fc=BLUE_L, ec=BLUE)
    box(ax, 55, 55, 20, 13, "Char n-gram TF-IDF\nchar_wb, 2–5 grams\nmin_df=2, sublinear\n(sparse, ~10^5 dims)", fc=AMBER_L, ec=AMBER)
    box(ax, 81, 66, 16, 14, "Horizontal\nconcatenation\n[emb ‖ tfidf]", fc=GREY_L, bold=True)
    box(ax, 81, 44, 16, 12, "Logistic\nregression\n(C=10, 10 classes)", fc=GREEN_L, ec=GREEN, bold=True)
    box(ax, 55, 26, 42, 11, "Temperature scaling  p = softmax(z / T),  T = 0.75\nfitted on out-of-fold decision values (NLL minimisation)", fc=RED_L, ec=RED)
    box(ax, 55, 10, 42, 10, "Calibrated probabilities\n→ top-1 label, confidence, top-3 list", fc=TEAL_L, ec=TEAL, bold=True)
    arrow(ax, (23, 69), (27.8, 69))
    arrow(ax, (48, 71), (54.8, 79), color=BLUE)
    arrow(ax, (48, 66), (54.8, 61), color=AMBER)
    arrow(ax, (75, 80), (80.8, 74), color=BLUE)
    arrow(ax, (75, 61), (80.8, 70), color=AMBER)
    arrow(ax, (89, 66), (89, 56.2))
    arrow(ax, (89, 44), (89, 37.2))
    arrow(ax, (76, 26), (76, 20.2))
    ax.text(35, 44, "Same encoder is reused by the\nretriever, so the query is encoded once.",
            fontsize=8.2, color=MUTED, ha="center")
    save(fig, "03_intent_model.png")


# ---------------------------------------------------------------- 4. retrieval
def retrieval_arch():
    fig, ax = canvas(12.5, 7.5)
    ax.text(50, 95, "Hybrid Retrieval Architecture", ha="center", fontsize=14, fontweight="bold")
    box(ax, 3, 66, 18, 12, "Query text +\n384-d embedding", fc=BLUE_L, ec=BLUE, bold=True)
    box(ax, 27, 76, 24, 10, "FAISS IndexFlatIP\n124,987 vectors, top-200\n(cosine = inner product)", fc=AMBER_L, ec=AMBER)
    box(ax, 27, 56, 24, 11, "TF-IDF matrix\nword 1–2 grams, 124,987 rows\ntop-200 by cosine", fc=AMBER_L, ec=AMBER)
    box(ax, 56, 66, 18, 12, "Candidate union\n(up to 400 ids)", fc=GREY_L, bold=True)
    box(ax, 56, 49, 18, 11, "Informative-answer\nmask (removes\ncall-log notes)", fc=RED_L, ec=RED)
    box(ax, 56, 31, 18, 12, "Fused score\nα·dense + (1−α)·tfidf\nα = 0.40 (held-out F1)", fc=GREEN_L, ec=GREEN, bold=True)
    box(ax, 27, 31, 24, 12, "Topic-aware re-ranking\ncandidate questions are classified;\nsame-topic answers first", fc=PURPLE_L, ec=PURPLE)
    box(ax, 27, 14, 24, 10, "Top-3 candidates\n(dense similarity kept for\nthe gate threshold)", fc=TEAL_L, ec=TEAL)
    box(ax, 56, 14, 18, 10, "Advisor memory\n(human answers since\ndeployment)", fc=BLUE_L, ec=BLUE)
    arrow(ax, (21, 72), (26.8, 79), color=BLUE)
    arrow(ax, (21, 69), (26.8, 61), color=AMBER)
    arrow(ax, (51, 81), (55.8, 74), color=AMBER)
    arrow(ax, (51, 61), (55.8, 70), color=AMBER)
    arrow(ax, (65, 66), (65, 60.2))
    arrow(ax, (65, 49), (65, 43.2))
    arrow(ax, (56, 36), (51.2, 36), color=GREEN)
    arrow(ax, (39, 31), (39, 24.2), color=PURPLE)
    arrow(ax, (65, 24), (65, 55.8), color=TEAL, ls="--")
    ax.text(66.5, 26, "merged into the\ncandidate pool", fontsize=7.6, color=TEAL, va="bottom")
    save(fig, "04_retrieval_architecture.png")


# ---------------------------------------------------------------- 5. gate flow
def gate_flow():
    fig, ax = canvas(11, 10.5)
    ax.text(50, 97, "Safety Gate Decision Logic (order of checks in code)", ha="center", fontsize=14,
            fontweight="bold")
    ys = [88, 78, 68, 58, 48, 38, 28]
    checks = [
        ("Predicted type in HIGH_STAKES_TYPES?\nPlant Protection · Nutrient Management · Government Scheme", "high_stakes"),
        ("Predicted type in TIME_SENSITIVE_TYPES?\nWeather · Market Price", "time_sensitive"),
        ("Intent confidence < 0.70?", "low_intent"),
        ("Dense retrieval similarity < 0.55?", "low_retrieval"),
        ("Retrieved question's predicted topic ≠ query type?", "topic_mismatch"),
        ("Retrieved answer is a call-log note (not advice)?", "uninformative_answer"),
        ("Answer contains dosage or chemical instruction?", "risky_answer"),
    ]
    for (t, code), y in zip(checks, ys):
        box(ax, 15, y - 6, 54, 6.2, t, fc="#ffffff", ec=INK, fs=8.3)
        box(ax, 76, y - 5.2, 18, 4.6, f"ESCALATE\n{code}", fc=RED_L, ec=RED, fs=7.8, bold=True)
        arrow(ax, (69.2, y - 1.4), (75.8, y - 1.4), color=RED)
        ax.text(71.5, y + 0.4, "YES", fontsize=7, color=RED, ha="center")
        if y != ys[-1]:
            arrow(ax, (42, y - 6), (42, ys[ys.index(y) + 1] + 0.2), color=GREEN)
            ax.text(44, y - 7.6, "no", fontsize=7, color=GREEN)
    box(ax, 22, 7, 46, 6.5, "AUTO_ANSWER\nreturn retrieved text + confidence + similarity", fc=GREEN_L, ec=GREEN, bold=True)
    arrow(ax, (42, 22), (45, 13.7), color=GREEN)
    ax.text(50, 2.5, "Every escalation carries a machine-readable reason code shown in the UI and stored in SQLite.",
            fontsize=8.2, color=MUTED, ha="center")
    save(fig, "05_safety_gate.png")


# ---------------------------------------------------------------- 6. provenance
def provenance():
    fig, ax = canvas(13, 8.2)
    ax.text(50, 96, "Data and Artefact Provenance", ha="center", fontsize=14, fontweight="bold")
    # real chain
    box(ax, 2, 76, 30, 12, "data.gov.in / IndiaAI\nKCC transcript catalogue\n(Open Government Licence – India)", fc=GREEN_L, ec=GREEN)
    box(ax, 36, 76, 28, 12, "Kaggle mirror by D. Koushik\n“Farmers Call Query (KCC) Data”\n2 columns: questions, answers", fc=GREEN_L, ec=GREEN)
    box(ax, 68, 76, 30, 12, "data/raw/kcc_mapping_full.csv\n139,542 rows · no state, date,\nlanguage, or audio metadata", fc=GREEN_L, ec=GREEN)
    arrow(ax, (32, 82), (35.8, 82), color=GREEN)
    arrow(ax, (64, 82), (67.8, 82), color=GREEN)
    ax.text(50, 71.5, "REAL-WORLD ADVISOR LOG TEXT (written summaries, not verbatim audio)", fontsize=8.5,
            color=GREEN, ha="center", fontweight="bold")

    box(ax, 2, 54, 22, 12, "Cleaning + weak labels\n138,875 rows\ntrain/val/test 80/10/10", fc=BLUE_L, ec=BLUE)
    box(ax, 27, 54, 22, 12, "Stratified sample\n3,496 queries\n(3,000 train / 496 test)", fc=BLUE_L, ec=BLUE)
    box(ax, 52, 54, 22, 12, "LLM-assisted labels\nClaude, written guidelines\nno human verification", fc=AMBER_L, ec=AMBER)
    box(ax, 77, 54, 21, 12, "Silver test set\n496 meaning-labelled\nheld-out corpus queries", fc=AMBER_L, ec=AMBER)
    arrow(ax, (24, 60), (26.8, 60), color=BLUE)
    arrow(ax, (49, 60), (51.8, 60), color=BLUE)
    arrow(ax, (74, 60), (76.8, 60), color=AMBER)

    box(ax, 2, 34, 22, 12, "Spoken-style augmentation\n400 train queries × 3 registers\n= 1,200 LLM paraphrases", fc=AMBER_L, ec=AMBER)
    box(ax, 27, 34, 22, 12, "Independent gold set\n150 LLM-written queries\n(50 EN / 50 Hinglish / 50 HI)", fc=AMBER_L, ec=AMBER)
    box(ax, 52, 34, 22, 12, "Synthetic speech\nWindows SAPI TTS\n50 English gold queries", fc=AMBER_L, ec=AMBER)
    box(ax, 77, 34, 21, 12, "Human review sheet\n250 rows · EMPTY (0 labels)\nintended human check,\nnot performed", fc=RED_L, ec=RED)
    arrow(ax, (38, 54), (14, 46.2), color=AMBER)
    arrow(ax, (20, 34), (20, 21.4), color=PURPLE)
    arrow(ax, (40, 34), (40, 21.4), color=PURPLE)
    arrow(ax, (62, 34), (62, 21.4), color=PURPLE)
    ax.text(50, 3.5, "Amber = LLM-generated or synthetic component · Green = real KCC log text · Red = intended human check, not performed",
            fontsize=7.8, color=MUTED, ha="center")

    box(ax, 14, 10, 32, 11, "Models & indexes\nMiniLM + LR head · FAISS · TF-IDF", fc=PURPLE_L, ec=PURPLE, bold=True)
    box(ax, 52, 10, 34, 11, "results/*.json\nall reported metrics and splits", fc=PURPLE_L, ec=PURPLE, bold=True)
    arrow(ax, (28, 34), (28, 21.2), color=PURPLE)
    arrow(ax, (55, 34), (65, 21.2), color=PURPLE)
    save(fig, "06_data_provenance.png")


# ---------------------------------------------------------------- 7. Gantt
def gantt():
    phases = [
        ("Problem identification", 0, 2),
        ("Literature review", 2, 3),
        ("Corpus acquisition & audit", 4, 3),
        ("Data cleaning & splitting", 7, 2),
        ("Meaning-based annotation (LLM-assisted)", 8, 4),
        ("Weak baselines (TF-IDF / SVM / DistilBERT)", 10, 3),
        ("Intent model development & calibration", 12, 3),
        ("Retrieval (FAISS dense → hybrid)", 13, 3),
        ("Safety gate & ablation", 15, 2),
        ("Frontend / backend integration", 16, 3),
        ("Testing & evaluation (gold, CI, McNemar)", 18, 3),
        ("Documentation, paper & audit", 20, 4),
    ]
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    colors = [BLUE, BLUE, GREEN, GREEN, AMBER, AMBER, PURPLE, PURPLE, RED, TEAL, "#7c3aed", MUTED]
    for i, (name, start, dur) in enumerate(phases):
        y = len(phases) - i - 1
        ax.barh(y, dur, left=start, height=0.55, color=colors[i], alpha=0.85, edgecolor="white")
    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels([p[0] for p in reversed(phases)], fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Project week (illustrative — repository history contains one commit, so no real dates exist)")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_title("Illustrative Project Schedule (24 weeks)", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    save(fig, "07_gantt_chart.png")


# ---------------------------------------------------------------- results figures
def results_figures():
    # confusion matrix parsed from the audit run (recomputed in audit)
    labels = ["Animal Husbandry", "Cultural Practices", "Fisheries", "Government Scheme", "Market Price",
              "Nutrient Management", "Other", "Plant Protection", "Varietal Information", "Weather"]
    cm = np.array([
        [68, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 75, 0, 0, 0, 4, 4, 1, 2, 0],
        [0, 0, 36, 0, 0, 0, 0, 2, 0, 0],
        [1, 0, 1, 44, 0, 0, 2, 0, 0, 0],
        [1, 0, 0, 0, 8, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 60, 0, 0, 0, 0],
        [0, 0, 1, 1, 2, 2, 13, 0, 0, 0],
        [0, 2, 0, 1, 0, 0, 0, 110, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0, 30, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 21],
    ])
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = ["AH", "CP", "FI", "GS", "MP", "NM", "OT", "PP", "VI", "WE"]
    ax.set_xticklabels(short, fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 55 else INK)
    ax.set_xlabel("Predicted query type")
    ax.set_ylabel("Meaning-based label (silver test, n=496)")
    ax.set_title("Confusion Matrix of the Deployed Intent Model", fontsize=13, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    fig.tight_layout()
    save(fig, "08_confusion_matrix.png")

    # gold by style
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    styles = ["English", "Hinglish", "Hindi"]
    x = np.arange(3)
    w = 0.26
    series = [
        ("Deployed (augmented)", [94, 90, 86], BLUE),
        ("No spoken-style augmentation", [84, 30, 66], AMBER),
        ("Char n-gram SVM (augmented)", [88, 88, 86], GREEN),
    ]
    for i, (name, vals, c) in enumerate(series):
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name, color=c, alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}%", ha="center", fontsize=8.4)
    ax.set_xticks(x)
    ax.set_xticklabels(styles)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy on independent gold set (%)")
    ax.set_title("Multilingual Generalisation: Gold Set (n=50 per register)", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.6, loc="lower left")
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "09_gold_multilingual.png")

    # gate ablation
    se = json.loads((ROOT / "results/selective_eval.json").read_text())
    configs = ["1_thresholds_only", "2_+high_stakes_rule", "3_+time_sensitive_rule",
               "4_+answer_screening", "5_+topic_match(full)"]
    names = ["Thresholds\nonly", "+ high-\nstakes", "+ time-\nsensitive", "+ answer\nscreening", "+ topic\nmatch (full)"]
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    cov = [se["gate_ablation_silver"][c]["coverage"] * 100 for c in configs]
    vio = [se["gate_ablation_silver"][c]["violation_rate_among_auto"] * 100 for c in configs]
    x = np.arange(len(configs))
    ax.bar(x - 0.19, cov, 0.38, label="Auto-answered (coverage, %)", color=BLUE, alpha=0.9)
    ax.bar(x + 0.19, vio, 0.38, label="Policy violations among auto-answers (%)", color=RED, alpha=0.85)
    for i, (a, b) in enumerate(zip(cov, vio)):
        ax.text(i - 0.19, a + 1.5, f"{a:.1f}", ha="center", fontsize=8)
        ax.text(i + 0.19, b + 1.5, f"{b:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Percent")
    ax.set_title("Gate Ablation on the 496-Query Silver Test (reproduced values)", fontsize=12.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8.8)
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "10_gate_ablation.png")

    # hybrid alpha curve
    hy = json.loads((ROOT / "results/hybrid_retrieval.json").read_text())
    alphas = sorted(float(k) for k in hy["alphas"].keys())
    allv = [hy["alphas"][str(a)]["all"] for a in alphas]
    novv = [hy["alphas"][str(a)]["novel"] for a in alphas]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(alphas, allv, "o-", color=BLUE, label="All informative test queries (n=2,737)")
    ax.plot(alphas, novv, "s--", color=AMBER, label="Novel queries (dense sim < 0.80, n=143)")
    ax.axvline(0.4, color=MUTED, linestyle=":", linewidth=1)
    ax.text(0.41, 0.20, "selected α = 0.40", fontsize=8, color=MUTED)
    ax.set_xlabel("Fusion weight α  (α=1 is dense-only; α=0 is TF-IDF-only)")
    ax.set_ylabel("Answer token-F1 vs advisor answer")
    ax.set_title("Hybrid Retrieval Fusion Sweep", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "11_hybrid_alpha.png")

    # learning curve
    re_ = json.loads((ROOT / "results/research_eval.json").read_text())
    lc = re_["learning_curve"]
    ns = sorted(int(k) for k in lc.keys())
    sil = [lc[str(n)]["silver_mean"] * 100 for n in ns]
    sil_sd = [lc[str(n)]["silver_sd"] * 100 for n in ns]
    gen = [lc[str(n)]["gold_en_mean"] * 100 for n in ns]
    gen_sd = [lc[str(n)]["gold_en_sd"] * 100 for n in ns]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.errorbar(ns, sil, yerr=sil_sd, fmt="o-", color=BLUE, capsize=3, label="Silver test (corpus queries)")
    ax.errorbar(ns, gen, yerr=gen_sd, fmt="s--", color=AMBER, capsize=3, label="Gold English queries")
    ax.set_xlabel("Number of annotated training queries")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(40, 100)
    ax.set_title("Learning Curve (mean ± s.d., 3 seeds, no augmentation)", fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "12_learning_curve.png")

    # novelty bins
    nv = re_["novelty_silver"]
    keys = ["novel (<0.80)", "similar (0.80-0.90)", "near-duplicate (>=0.90)"]
    vals = [nv[k]["accuracy"] * 100 for k in keys]
    ns_ = [nv[k]["n"] for k in keys]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.bar(["Novel\n(<0.80)", "Similar\n(0.80–0.90)", "Near-duplicate\n(≥0.90)"], vals,
                  color=[RED, AMBER, GREEN], alpha=0.9)
    for b, v, n in zip(bars, vals, ns_):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accuracy on silver test (%)")
    ax.set_title("Accuracy by Similarity to the Training Set", fontsize=12.5, fontweight="bold")
    ax.grid(axis="y", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "13_novelty.png")

    # corpus class distribution
    dist = pd.Series({"Plant Protection": 53813, "Other": 41471, "Nutrient Management": 19871,
                      "Varietal Information": 6428, "Government Scheme": 6404, "Animal Husbandry": 4776,
                      "Fisheries": 3139, "Cultural Practices": 1847, "Market Price": 679, "Weather": 447})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.barh(dist.index[::-1], dist.values[::-1], color=BLUE, alpha=0.85)
    for i, v in enumerate(dist.values[::-1]):
        ax.text(v + 800, i, f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Queries in cleaned corpus (138,875)")
    ax.set_title("Weak-Label Distribution Used to Stratify the Splits", fontsize=12.5, fontweight="bold")
    ax.grid(axis="x", color="#e5e7eb")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "14_corpus_distribution.png")


if __name__ == "__main__":
    architecture()
    workflow()
    model_arch()
    retrieval_arch()
    gate_flow()
    provenance()
    gantt()
    results_figures()
    print("all diagrams written to", OUT)
