"""Generate every number, table and figure used in the paper from results/*.json.

Writes paper/generated/numbers.tex (one \\newcommand per number), table_*.tex and
paper/figures/*.pdf. Missing results become "n/a" so the paper still compiles
while experiments are running; nothing here is typed by hand.

Usage: .venv/Scripts/python.exe scripts/19_make_paper_assets.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"
GEN.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


RE, SE, HY, VO = load("research_eval.json"), load("selective_eval.json"), load("hybrid_retrieval.json"), load("voice_eval.json")
CAL, ANN, HUM = load("calibration.json"), load("annotation_summary.json"), load("human_agreement.json")
SCR, MS = load("answer_screening.json"), load("model_selection.json")

NUM = {}


def pc(x, d=1):
    return "n/a" if x is None else f"{x * 100:.{d}f}\\%"


def setn(name, val):
    NUM[name] = val


PROD = "production (emb+char, aug)"
if RE:
    m = RE["main"]
    for key, nm in [("silver", "silverAcc"), ("gold_all", "goldAcc"), ("gold_en", "goldEn"),
                    ("gold_hinglish", "goldHing"), ("gold_hi", "goldHi")]:
        setn(nm, pc(m[PROD][key]["acc"]))
        lo, hi = m[PROD][key]["ci95"]
        setn(nm + "CI", f"{lo*100:.1f}--{hi*100:.1f}\\%")
    setn("hingPert", pc(m[PROD]["hinglish_perturbed"]["acc"]))
    a = RE["agreement_vs_accuracy"]
    setn("oldAgreeKw", pc(a["old_distilbert_agreement_with_keyword_labels"]))
    setn("oldAccMeaning", pc(a["old_distilbert_accuracy_vs_meaning_labels_silver"]))
    setn("prodAgreeKw", pc(a["production_agreement_with_keyword_labels"]))
    c = RE["corpus_audit"]
    setn("corpusRows", f"{c['corpus_rows']:,}".replace(",", "{,}"))
    setn("nearDup", pc(c["weak_test_near_duplicate_share_ge_0.95"], 0))
    setn("retNearDup", pc(c["retrieval_category_match_near_duplicate"], 0))
    setn("retNovel", pc(c["retrieval_category_match_novel"], 0))
    setn("callLogShare", pc(c["share_answers_that_are_call_log_notes"]))
    g = c["geography"]
    setn("haryanaMentions", str(g["mention_haryana_places"]))
    setn("assamMentions", f"{g['mention_assam_places_or_terms']:,}".replace(",", "{,}"))
    setn("goldN", "150")
    for k in ["word TF-IDF + SVM", "char TF-IDF + SVM", "embedding + LR", "keyword rules (original)"]:
        tag = {"word TF-IDF + SVM": "wordSvm", "char TF-IDF + SVM": "charSvm",
               "embedding + LR": "embLr", "keyword rules (original)": "kwRules"}[k]
        setn(tag + "Silver", pc(m[k]["silver"]["acc"]))
        setn(tag + "Gold", pc(m[k]["gold_all"]["acc"]))
if RE:
    mc = RE["mcnemar_vs_production"]
    for key, tag in [("char TF-IDF + SVM", "Char"), ("word TF-IDF + SVM", "Word"), ("embedding + LR", "Emb")]:
        setn("mc" + tag + "Silver", f"{mc[key]['silver']['p_value']:.3f}")
        setn("mc" + tag + "Gold", f"{mc[key]['gold_all']['p_value']:.3f}")
    m_ = RE["main"]
    setn("charSvmGoldHi", pc(m_["char TF-IDF + SVM"]["gold_hi"]["acc"]))
    setn("charSvmHingPert", pc(m_["char TF-IDF + SVM"]["hinglish_perturbed"]["acc"]))
    ab_ = RE["ablation"]
    setn("noAugGold", pc(ab_["embedding + char, no augmentation"]["gold_all"]))
    setn("noAugSilver", pc(ab_["embedding + char, no augmentation"]["silver"]))
    setn("noAugHing", pc(ab_["embedding + char, no augmentation"]["gold_hinglish"], 0))
    nv_ = RE["novelty_silver"]
    setn("silverNovel", pc(nv_["novel (<0.80)"]["accuracy"])); setn("silverDup", pc(nv_["near-duplicate (>=0.90)"]["accuracy"]))
    if "char n-grams only, no augmentation" in ab_:
        setn("charNoAugHi", pc(ab_["char n-grams only, no augmentation"]["gold_hi"], 0))
    setn("embNoAugHi", pc(ab_["embedding only, no augmentation"]["gold_hi"], 0))
BL = load("baselines.json")
if BL:
    o = BL["intent_weak_label_test"]
    setn("tfidfOldTest", pc(o["tfidf_linearsvc"]["all"]["accuracy"]))
    setn("bertOldTest", pc(o["finetuned_distilbert"]["all"]["accuracy"]))
if ANN:
    setn("annN", f"{ANN['n']:,}".replace(",", "{,}"))
    setn("kwAgree", pc(ANN["keyword_vs_semantic_agreement"]))
if SE:
    sd, gd = SE["silver_default"], SE["gold_default"]
    setn("selCovSilver", pc(sd["coverage"], 0))
    setn("selAccSilver", pc(sd["selective_accuracy"]))
    setn("selCovGold", pc(gd["coverage"], 0))
    setn("selAccGold", pc(gd["selective_accuracy"]))
    ab = SE.get("gate_ablation_silver", {})
    if ab:
        t, f = ab["1_thresholds_only"], ab["5_+topic_match(full)"]
        setn("covThrOnly", pc(t["coverage"], 0)); setn("violThrOnly", pc(t["violation_rate_among_auto"], 0))
        setn("covFull", pc(f["coverage"], 0)); setn("violFull", pc(f["violation_rate_among_auto"], 0))
        o4 = ab["4_+answer_screening"]
        setn("offTopicBefore", pc(o4["offtopic_answer_rate_among_auto"], 0))
        setn("offTopicAfter", pc(f["offtopic_answer_rate_among_auto"], 0))
if SCR:
    setn("scrN", f"{SCR['n']:,}".replace(",", "{,}"))
    setn("scrCovOff", pc(SCR["coverage_without_screening"])); setn("scrCovOn", pc(SCR["coverage_with_screening"]))
    setn("scrRiskyOff", pc(SCR["auto_answers_with_dosage_or_chemical_text_without_screening"], 0))
if CAL:
    setn("calT", f"{CAL['temperature']:.2f}")
    setn("eceOofBefore", pc(CAL["ece_oof_uncalibrated"])); setn("eceOofAfter", pc(CAL["ece_oof_calibrated"]))
    setn("eceSilBefore", pc(CAL["ece_silver_uncalibrated"])); setn("eceSilAfter", pc(CAL["ece_silver_calibrated"]))
if HY:
    al = HY["alphas"]; ch = str(HY["chosen_alpha"])
    setn("alphaChosen", ch); setn("hyN", str(HY["n"])); setn("hyNovelN", str(HY["n_novel"]))
    setn("fDense", f"{al['1.0']['all']:.3f}"); setn("fHybrid", f"{al[ch]['all']:.3f}")
    setn("fDenseNovel", f"{al['1.0']['novel']:.3f}"); setn("fHybridNovel", f"{al[ch]['novel']:.3f}")
    setn("fTfidf", f"{al['0.0']['all']:.3f}")
if VO:
    setn("werMean", pc(VO["mean_wer"])); setn("voiceN", str(VO["n"]))
    setn("voiceTextAcc", pc(VO["intent_accuracy_typed_text"], 0)); setn("voiceAsrAcc", pc(VO["intent_accuracy_after_asr"], 0))
if HUM:
    ann = list(HUM["annotators"].values())
    setn("humKappa", f"{np.mean([a['cohen_kappa_vs_llm'] for a in ann]):.2f}")
    setn("humAcc", pc(float(np.mean([a["accuracy_vs_llm"] for a in ann]))))
    if "human_vs_human" in HUM:
        setn("hhKappa", f"{HUM['human_vs_human']['cohen_kappa']:.2f}")
else:
    setn("humKappa", "n/a"); setn("humAcc", "n/a"); setn("hhKappa", "n/a")

import re

main_tex = ROOT / "paper" / "main.tex"
LATEX_BUILTINS = {"date", "centering", "small", "maketitle", "item", "emph", "textbf", "texttt"}
used = (set(re.findall(r"\\([A-Za-z]+)\{\}", main_tex.read_text(encoding="utf-8"))) - LATEX_BUILTINS) if main_tex.exists() else set()
with open(GEN / "numbers.tex", "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED by scripts/19_make_paper_assets.py -- do not edit\n")
    for k in sorted(used | set(NUM)):
        f.write(f"\\newcommand{{\\{k}}}{{{NUM.get(k, 'n/a')}}}\n")
missing = sorted(used - set(NUM))
print(f"wrote {len(NUM)} numbers; {len(missing)} still n/a: {missing}")


# ------------------------------------------------------------------ tables
def tex_escape(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


if RE:
    rows = [("keyword rules (original)", "Keyword rules (original labeller)"),
            ("DistilBERT on keyword labels (original)", "DistilBERT trained on keyword labels"),
            ("word TF-IDF + SVM", "Word TF-IDF + SVM"), ("char TF-IDF + SVM", "Character TF-IDF + SVM"),
            ("embedding + LR", "Sentence embedding + LR"), (PROD, "Embedding + char TF-IDF + LR (ours)")]
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "System & Silver & Gold & EN & Hinglish & Hindi & Hing.$^{*}$ \\\\", "\\midrule"]
    for k, label in rows:
        r = RE["main"][k]
        hp = r["hinglish_perturbed"]
        cells = [f"{r[s]['acc']*100:.1f}" for s in ["silver", "gold_all", "gold_en", "gold_hinglish", "gold_hi"]]
        cells.append(f"{hp['acc']*100:.1f}" if hp else "--")
        row = f"{tex_escape(label)} & " + " & ".join(cells) + " \\\\"
        lines.append(("\\rowcolor{gray!15} " if k == PROD else "") + row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_main.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Variant & Silver & Gold & EN & Hinglish & Hindi \\\\", "\\midrule"]
    for k, r in RE["ablation"].items():
        lines.append(f"{tex_escape(k)} & " + " & ".join(f"{r[s]*100:.1f}" for s in ["silver", "gold_all", "gold_en", "gold_hinglish", "gold_hi"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_ablation.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Test queries (silver) & $n$ & Accuracy \\\\", "\\midrule"]
    for k, r in RE["novelty_silver"].items():
        lines.append(f"{tex_escape(k)} & {r['n']} & {r['accuracy']*100:.1f}\\% \\\\" if r["accuracy"] is not None else f"{tex_escape(k)} & 0 & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_novelty.tex").write_text("\n".join(lines), encoding="utf-8")

if SE and SE.get("gate_ablation_silver"):
    names = {"1_thresholds_only": "Thresholds only", "2_+high_stakes_rule": "+ high-stakes rule",
             "3_+time_sensitive_rule": "+ time-sensitive rule", "4_+answer_screening": "+ answer screening", "5_+topic_match(full)": "+ topic match (full)"}
    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{3}{c}{Silver (held-out calls)} & \\multicolumn{3}{c}{Gold (independent)} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Gate & Cover. & Acc. & Viol. & Cover. & Acc. & Viol. \\\\", "\\midrule"]
    for k, label in names.items():
        s, g = SE["gate_ablation_silver"][k], SE["gate_ablation_gold"][k]
        f = lambda v: "--" if v is None else f"{v*100:.1f}"
        lines.append(f"{label} & {f(s['coverage'])} & {f(s['selective_accuracy'])} & {f(s['violation_rate_among_auto'])} & "
                     f"{f(g['coverage'])} & {f(g['selective_accuracy'])} & {f(g['violation_rate_among_auto'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_gate.tex").write_text("\n".join(lines), encoding="utf-8")

if HY:
    lines = ["\\begin{tabular}{rrr}", "\\toprule", "TF-IDF weight $1-\\alpha$ & All & Novel \\\\", "\\midrule"]
    for a, v in sorted(HY["alphas"].items(), key=lambda kv: float(kv[0])):
        lines.append(f"{1 - float(a):.1f} & {v['all']:.3f} & {v['novel']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "table_hybrid.tex").write_text("\n".join(lines), encoding="utf-8")

# ----------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

if RE:
    lc = RE["learning_curve"]
    ns = [int(k) for k in lc]
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for key, lab in [("silver", "held-out calls"), ("gold_en", "gold (English)")]:
        mu = np.array([lc[str(n)][f"{key}_mean"] for n in ns]) * 100
        sd = np.array([lc[str(n)][f"{key}_sd"] for n in ns]) * 100
        ax.errorbar(ns, mu, yerr=sd, marker="o", capsize=2, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("annotated training queries"); ax.set_ylabel("accuracy (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "learning_curve.pdf"); plt.close(fig)

if SE:
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for key, lab in [("silver_sweep", "held-out calls"), ("gold_sweep", "gold")]:
        pts = [(r["coverage"] * 100, r["selective_accuracy"] * 100) for r in SE[key] if r["selective_accuracy"] is not None and r["coverage"] > 0]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=10, alpha=0.6, label=lab)
    ax.set_xlabel("auto-answered (%)"); ax.set_ylabel("accuracy of auto-answered (%)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "selective.pdf"); plt.close(fig)

conf = RES / "weak_vs_semantic_confusion.csv"
if conf.exists():
    ct = pd.read_csv(conf, index_col=0)
    ab = {"Animal Husbandry": "AH", "Cultural Practices": "CP", "Fisheries": "FI", "Government Scheme": "GS", "Market Price": "MP",
          "Nutrient Management": "NM", "Other": "OT", "Plant Protection": "PP", "Varietal Information": "VI", "Weather": "WE"}
    ct = ct.rename(index=ab, columns=ab)
    norm = ct.div(ct.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(norm.values, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(norm.columns))); ax.set_xticklabels(norm.columns, rotation=90)
    ax.set_yticks(range(len(norm.index))); ax.set_yticklabels(norm.index)
    ax.set_xlabel("label by meaning"); ax.set_ylabel("keyword-rule label")
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(FIG / "label_confusion.pdf"); plt.close(fig)
print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
print("tables:", sorted(p.name for p in GEN.glob("table_*.tex")))
