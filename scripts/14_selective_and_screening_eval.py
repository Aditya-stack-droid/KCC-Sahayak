"""Evaluate the deployed pipeline the way a triage system should be judged.

A. Selective accuracy vs coverage: as the gate tightens, what fraction of
   queries is still auto-answered (coverage) and how correct is that subset?
   Scored on the held-out silver test (in-distribution, meaning-based labels,
   with advisor answers) and the gold set (independent phrasing, 3 languages).
   Correctness = predicted intent equals the annotated label. For silver we also
   report answer-level F1 against the advisor's real answer.

B. Answer screening: of the queries the gate would auto-answer WITHOUT looking at
   the retrieved text, what fraction would have sent a dosage / chemical
   instruction? (Measured on the 3,000-query held-out subset.)

Outputs: results/selective_eval.json, results/answer_screening.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from answer_quality import is_informative  # noqa: E402
from confidence_gate import GateThresholds, answer_risk, decide  # noqa: E402
from glossary import HIGH_STAKES_TYPES, TIME_SENSITIVE_TYPES  # noqa: E402
from pipeline import KCCSahayakPipeline  # noqa: E402
from query_match import coverage_problem  # noqa: E402

RES = ROOT / "results"


def token_f1(a, b):
    ta, tb = re.findall(r"\w+", str(a).lower()), re.findall(r"\w+", str(b).lower())
    if not ta or not tb:
        return 0.0
    n = sum((Counter(ta) & Counter(tb)).values())
    if n == 0:
        return 0.0
    p, r = n / len(ta), n / len(tb)
    return 2 * p * r / (p + r)


def probe(pipe, texts):
    rows = []
    for t in texts:
        emb = pipe._embed(t)
        label, conf, _ = pipe.classify_intent(t, emb)
        cands = pipe.retrieve(t, emb, k=3, query_type=label)
        top = cands[0] if cands else {"similarity": 0.0, "answer": "", "type": None}
        rows.append({"pred": label, "conf": conf, "sim": top["similarity"], "answer": top["answer"],
                     "rtype": top.get("type"),
                     "cov": coverage_problem(t, top.get("matched_question", ""), top.get("answer", "")) if top.get("matched_question") else None})
    return pd.DataFrame(rows)


def auto_mask(df, th):
    """Boolean array: would the gate auto-answer each row under thresholds `th`?"""
    return np.array([decide(p, c, s, th, a, retrieved_type=rt, coverage_problem=cv).action == "AUTO_ANSWER"
                     for p, c, s, a, rt, cv in zip(df["pred"], df["conf"], df["sim"], df["answer"], df["rtype"],
                                                   [c if isinstance(c, str) else None for c in df["cov"]])])


def main():
    pipe = KCCSahayakPipeline()

    ann = pd.read_csv(ROOT / "data/annotation/annotated.csv")
    silver = ann[ann["pool"] == "test"].reset_index(drop=True)
    gold = pd.read_csv(ROOT / "data/gold/gold_queries.csv")

    print("probing silver ...")
    S = pd.concat([silver, probe(pipe, silver["text"])], axis=1)
    S["correct"] = S["pred"] == S["label"]
    S["f1"] = [token_f1(a, b) for a, b in zip(S["answer"], S["answers"])]
    print("probing gold ...")
    G = pd.concat([gold, probe(pipe, gold["text"])], axis=1)
    G["correct"] = G["pred"] == G["label"]

    def sweep(df, with_f1=False):
        rows = []
        for conf in np.arange(0.30, 0.96, 0.05):
            for sim in np.arange(0.30, 0.91, 0.10):
                th = GateThresholds(round(float(conf), 2), round(float(sim), 2))
                dec = auto_mask(df, th)
                r = {"conf": th.intent_confidence, "sim": th.retrieval_similarity,
                     "coverage": round(float(dec.mean()), 4),
                     "selective_accuracy": round(float(df.loc[dec, "correct"].mean()), 4) if dec.any() else None}
                if with_f1:
                    r["answer_f1"] = round(float(df.loc[dec, "f1"].mean()), 4) if dec.any() else None
                rows.append(r)
        return rows

    default = GateThresholds()

    def at_default(df, with_f1=False):
        dec = auto_mask(df, default)
        out = {"n": int(len(df)), "coverage": round(float(dec.mean()), 4),
               "selective_accuracy": round(float(df.loc[dec, "correct"].mean()), 4) if dec.any() else None,
               "overall_accuracy": round(float(df["correct"].mean()), 4)}
        if with_f1:
            out["answer_f1_on_auto"] = round(float(df.loc[dec, "f1"].mean()), 4) if dec.any() else None
        return out

    result = {
        "default_thresholds": {"intent_confidence": default.intent_confidence,
                               "retrieval_similarity": default.retrieval_similarity},
        "silver_default": at_default(S, True),
        "gold_default": at_default(G),
        "gold_default_by_style": {st: at_default(G[G["style"] == st].reset_index(drop=True)) for st in ["en", "hinglish", "hi"]},
        "silver_sweep": sweep(S, True),
        "gold_sweep": sweep(G),
    }
    # ---- gate ablation: what does each rule buy?
    # A "violation" is an auto-answer the policy says should not have been sent:
    #   (i) the query truly belongs to a high-stakes / time-sensitive category
    #       (the classifier mis-routed it), or
    #   (ii) the text sent contains dosage/chemical advice or is only a log note.
    risky_cats = HIGH_STAKES_TYPES | TIME_SENSITIVE_TYPES
    configs = {
        "1_thresholds_only": GateThresholds(block_high_stakes=False, block_time_sensitive=False, screen_answer=False),
        "2_+high_stakes_rule": GateThresholds(block_high_stakes=True, block_time_sensitive=False, screen_answer=False),
        "3_+time_sensitive_rule": GateThresholds(block_high_stakes=True, block_time_sensitive=True, screen_answer=False,
                                                 require_topic_match=False),
        "4_+answer_screening": GateThresholds(require_topic_match=False),
        "5_+topic_match(full)": GateThresholds(),  # topic + crop/comparison coverage
    }
    for _th in list(configs.values())[:2]:
        _th.require_topic_match = False

    def ablate(df):
        out = {}
        for name, th in configs.items():
            dec = auto_mask(df, th)
            offtopic = (df["rtype"] != df["label"]).to_numpy()
            viol_cat = df["label"].isin(risky_cats).to_numpy()
            viol_txt = np.array([(answer_risk(a) is not None) or (not is_informative(a)) for a in df["answer"]])
            viol = viol_cat | viol_txt
            out[name] = {
                "coverage": round(float(dec.mean()), 4),
                "selective_accuracy": round(float(df.loc[dec, "correct"].mean()), 4) if dec.any() else None,
                "violation_rate_among_auto": round(float(viol[dec].mean()), 4) if dec.any() else None,
                "offtopic_answer_rate_among_auto": round(float(offtopic[dec].mean()), 4) if dec.any() else None,
                "n_auto": int(dec.sum()),
            }
        return out

    result["gate_ablation_silver"] = ablate(S)
    result["gate_ablation_gold"] = ablate(G)
    print("gate ablation (silver):")
    for k, v in result["gate_ablation_silver"].items():
        print(f"  {k:28s} {v}")
    print("gate ablation (gold):")
    for k, v in result["gate_ablation_gold"].items():
        print(f"  {k:28s} {v}")
    (RES / "selective_eval.json").write_text(json.dumps(result, indent=2))
    print("silver @default:", result["silver_default"])
    print("gold   @default:", result["gold_default"])
    print("gold by style  :", result["gold_default_by_style"])

    # ---- B. answer screening on the 3,000 held-out test subset
    preds = pd.read_csv(RES / "test_predictions.csv")
    test = pd.read_csv(ROOT / "data/processed/test.csv").dropna(subset=["questions", "answers"])
    test = test[test["qid"].isin(preds["qid"])].drop_duplicates("qid").reset_index(drop=True)
    print("probing 3,000 test queries ...")
    T = pd.concat([test[["questions"]], probe(pipe, test["questions"])], axis=1)
    auto_off = auto_mask(T, GateThresholds(screen_answer=False, require_topic_match=False))
    auto_on = auto_mask(T, GateThresholds(require_topic_match=False))
    risky = np.array([answer_risk(a) is not None for a in T["answer"]])
    scr = {
        "n": int(len(T)),
        "coverage_without_screening": round(float(auto_off.mean()), 4),
        "coverage_with_screening": round(float(auto_on.mean()), 4),
        "auto_answers_with_dosage_or_chemical_text_without_screening": round(float(risky[auto_off].mean()), 4) if auto_off.any() else None,
        "auto_answers_with_dosage_or_chemical_text_with_screening": round(float(risky[auto_on].mean()), 4) if auto_on.any() else None,
    }
    (RES / "answer_screening.json").write_text(json.dumps(scr, indent=2))
    print("answer screening:", scr)


if __name__ == "__main__":
    main()
