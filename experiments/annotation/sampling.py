"""Build the 300-query English annotation pilot sample.

Data provenance
---------------
Everything is drawn from the **existing** corpus annotations
(``data/annotation/annotated.csv``, 3,496 rows sampled from
``data/processed/full_labeled.csv``).  Nothing is downloaded, no model is
trained, and no original file is modified.  The existing LLM labels are used
*only* to stratify the sample and are hidden from annotators.

Strata (declared before selection)
----------------------------------
``potentially_unsafe``  query mentions a pesticide/fertiliser, a dose unit or
                        a spray instruction (the Phase 1/3 risk vocabulary).
``insufficient_context`` query names a symptom or pest but no crop token, so an
                        answer would need the crop to be safe.
``ambiguous``           very short query, or a short ``Other``-labelled query.
``rare_intent``         Weather, Market Price, Fisheries, Cultural Practices.
``novel``               no corpus question at >= 0.80 cosine similarity.
``near_duplicate``      at least one other corpus row in the same Batch A
                        0.95-similarity group.
``common``              a frequent category (also the fallback filler).

Selection is greedy within each stratum, in that priority order, with two
dedupe rules: normalised question text is unique, and no two selected queries
have cosine similarity >= 0.98.  A query belongs to exactly one stratum, so
the realised counts are reported honestly (including shortfalls).

Near-duplicate and novelty signals reuse Batch A artefacts, which are read-only:
``experiments/grouped_evaluation/artifacts/corpus_embeddings.npy``,
``emb_annotated_texts.npy`` and ``corpus_split.csv`` (group ids at 0.95).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
ANN_CSV = ROOT / "data" / "annotation" / "annotated.csv"
BATCH_A = ROOT / "experiments" / "grouped_evaluation" / "artifacts"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_N = 300
DEFAULT_SEED = 0
NEAR_DUPLICATE = 0.95
SIMILAR = 0.80
DEDUPE_SIM = 0.98

STRATA_ORDER: tuple[str, ...] = (
    "potentially_unsafe",
    "insufficient_context",
    "ambiguous",
    "rare_intent",
    "novel",
    "near_duplicate",
    "common",
)

DEFAULT_QUOTAS: dict[str, int] = {
    "potentially_unsafe": 25,
    "insufficient_context": 35,
    "ambiguous": 35,
    "rare_intent": 55,
    "novel": 30,
    "near_duplicate": 40,
    "common": 80,
}

RARE_LABELS = {"Weather", "Market Price", "Fisheries", "Cultural Practices"}
COMMON_LABELS = {"Plant Protection", "Nutrient Management", "Animal Husbandry", "Government Scheme", "Varietal Information"}

RISK_RE = re.compile(
    r"\b(?:urea|dap|ssp|mop|npk|potash|zinc|borax|sulphur|sulfur|insecticide|pesticide|"
    r"fungicide|herbicide|weedicide|acaricide|chlorpyrifos|imidacloprid|cypermethrin|"
    r"mancozeb|carbendazim|copper|bordeaux|rogor|dimethoate|monocrotophos|endosulfan|"
    r"butachlor|glyphosate|neem|spray|spraying|dusting|dose|dosage|poison|medicine)\b",
    re.IGNORECASE,
)
DOSE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:ml|mls|gm|gms|g|kg|kgs|lit|litre|liter|l|q|quintal|ppm|%)\b|@\s*\d",
    re.IGNORECASE,
)
SYMPTOM_RE = re.compile(
    r"\b(?:pest|insect|disease|wilt|wilting|spot|spots|yellow|yellowing|curl|curling|rot|"
    r"blight|attack|damage|dying|dried|holes|fungus|fungal|caterpillar|aphid|thrips|borer|"
    r"mite|weevil|smut|rust|mildew|sucking|deficiency)\b",
    re.IGNORECASE,
)
CROP_WORDS = {
    "rice", "paddy", "wheat", "maize", "tomato", "brinjal", "potato", "cotton", "mustard",
    "sugarcane", "chilli", "chili", "onion", "okra", "mango", "banana", "pulse", "gram",
    "lentil", "soybean", "groundnut", "coconut", "tea", "ginger", "turmeric", "cabbage",
    "cauliflower", "cucumber", "gourd", "papaya", "guava", "litchi", "orange", "apple",
    "grape", "arecanut", "jute", "sesame", "bajra", "jowar", "ragi", "barley", "pea",
    "bean", "cowpea", "blackgram", "greengram", "arhar", "linseed", "castor", "sunflower",
    "safflower", "niger", "fodder", "sugarcane",
}


# --------------------------------------------------------------------------- #
# Existing helpers (imported read-only, never modified)
# --------------------------------------------------------------------------- #
def _load_lang_module():
    path = ROOT / "src" / "lang.py"
    spec = importlib.util.spec_from_file_location("kcc_lang_phase2", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect_register(text: str) -> str:
    """Register detection, reusing the shipped ``src/lang.py`` heuristic."""
    return _load_lang_module().detect_register(text)


def _sha1(texts: Iterable[str]) -> str:
    h = hashlib.sha1()
    for text in texts:
        h.update(str(text).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def normalise_question(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).lower()).split())


# --------------------------------------------------------------------------- #
# Candidate pool and novelty features
# --------------------------------------------------------------------------- #
def _corpus_features() -> pd.DataFrame:
    """Group size and max non-self cosine similarity for each annotated row."""
    ann = pd.read_csv(ANN_CSV)
    ann["text"] = ann["text"].astype(str)

    cache_path = DATA / "sample_features.json"
    digest = _sha1(ann["text"])
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("sha1") == digest
            and len(cached.get("max_similarity", [])) == len(ann)
            and len(cached.get("group_size", [])) == len(ann)
        ):
            ann["max_similarity"] = cached["max_similarity"]
            ann["near_duplicate_neighbours"] = cached["near_duplicate_neighbours"]
            ann["corpus_group_size"] = cached["group_size"]
            return ann

    if not (BATCH_A / "corpus_embeddings.npy").exists():
        raise FileNotFoundError(
            "Batch A corpus embeddings are required for novelty stratification; "
            f"missing {BATCH_A / 'corpus_embeddings.npy'}"
        )
    corpus = np.load(BATCH_A / "corpus_embeddings.npy", mmap_mode="r")
    split = pd.read_csv(BATCH_A / "corpus_split.csv")
    group_counts = split.groupby("group_id").size()
    qid_rows: dict[Any, list[int]] = {
        qid: rows.tolist() for qid, rows in split.reset_index().groupby("qid")["index"]
    }
    qid_group_size = {
        qid: int(max(group_counts[g] for g in groups["group_id"]))
        for qid, groups in split.groupby("qid")
    }

    texts = ann["text"].tolist()
    emb_meta = json.loads((BATCH_A / "emb_annotated_texts.json").read_text(encoding="utf-8"))
    if emb_meta.get("sha1") != digest:
        raise RuntimeError(
            "annotated.csv no longer matches the Batch A annotation embeddings "
            f"(expected {emb_meta.get('sha1')}, got {digest}); recompute Phase 2 "
            "features explicitly instead of guessing."
        )
    emb = np.load(BATCH_A / "emb_annotated_texts.npy").astype("float32")

    max_sim = np.full(len(ann), -1.0, dtype="float32")
    neighbours = np.zeros(len(ann), dtype="int32")
    chunk = 256
    for start in range(0, len(ann), chunk):
        stop = min(start + chunk, len(ann))
        sims = emb[start:stop] @ np.asarray(corpus).T
        for offset, row in enumerate(range(start, stop)):
            qid = ann.iloc[row]["qid"]
            own = qid_rows.get(qid, [])
            if own:
                sims[offset, own] = -1.0
            max_sim[row] = float(sims[offset].max())
            neighbours[row] = int((sims[offset] >= NEAR_DUPLICATE).sum())

    ann["max_similarity"] = max_sim.round(6)
    ann["near_duplicate_neighbours"] = neighbours
    ann["corpus_group_size"] = [int(qid_group_size.get(q, 1)) for q in ann["qid"]]
    DATA.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "sha1": digest,
        "model": EMBED_MODEL,
        "source": "experiments/grouped_evaluation/artifacts/{corpus_embeddings.npy,emb_annotated_texts.npy,corpus_split.csv}",
        "near_duplicate_threshold": NEAR_DUPLICATE,
        "similar_threshold": SIMILAR,
        "excludes_same_qid_rows": True,
        "n": int(len(ann)),
        "max_similarity": ann["max_similarity"].tolist(),
        "near_duplicate_neighbours": ann["near_duplicate_neighbours"].tolist(),
        "group_size": ann["corpus_group_size"].tolist(),
    }, indent=2), encoding="utf-8")
    return ann


def build_pool() -> tuple[pd.DataFrame, dict]:
    """Return the eligible English candidate pool plus composition statistics."""
    ann = _corpus_features()
    total = len(ann)
    stats: dict[str, Any] = {"annotated_rows": total}

    questions = ann["text"].astype(str).str.strip()
    answers = ann["answers"].fillna("").astype(str).str.strip()
    keep = (questions.str.len().between(12, 300)) & (answers.str.len() >= 25)
    stats["dropped_length_or_empty_answer"] = int((~keep).sum())

    registers = questions.map(detect_register)
    english = registers == "en"
    stats["dropped_non_english"] = int((keep & ~english).sum())
    stats["register_counts_before_filter"] = registers.value_counts().to_dict()

    pool = ann.loc[keep & english].copy()
    pool["source_row"] = pool.index.to_numpy()
    pool = pool.reset_index(drop=True)
    pool["register"] = "en"
    pool["normalised"] = pool["text"].map(normalise_question)
    stats["eligible_pool"] = int(len(pool))
    return pool, stats


def _no_crop_token(text: str) -> bool:
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    return not (tokens & CROP_WORDS)


def assign_stratum(row: pd.Series) -> str:
    """Eligibility order; the first matching stratum owns the query."""
    text = str(row["text"])
    if RISK_RE.search(text) or DOSE_RE.search(text):
        return "potentially_unsafe"
    crops = row.get("crops")
    has_crops = isinstance(crops, str) and bool(crops.strip())
    if SYMPTOM_RE.search(text) and _no_crop_token(text) and not has_crops:
        return "insufficient_context"
    label = str(row["label"])
    if len(text) <= 45 or (label == "Other" and len(text) <= 100):
        return "ambiguous"
    if label in RARE_LABELS:
        return "rare_intent"
    if float(row["max_similarity"]) < SIMILAR:
        return "novel"
    if int(row["corpus_group_size"]) >= 2:
        return "near_duplicate"
    return "common"


def select_sample(pool: pd.DataFrame, n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                  quotas: dict[str, int] | None = None) -> tuple[pd.DataFrame, dict]:
    """Greedy stratified selection with text and embedding dedupe."""
    quotas = dict(quotas or DEFAULT_QUOTAS)
    if sum(quotas.values()) != n:
        raise ValueError(f"quotas sum to {sum(quotas.values())}, expected {n}")

    pool = pool.copy()
    pool["stratum"] = pool.apply(assign_stratum, axis=1)
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)

    emb = np.load(BATCH_A / "emb_annotated_texts.npy").astype("float32")
    pool_emb = emb[pool["source_row"].to_numpy()]

    selected: list[int] = []
    chosen: dict[str, list[int]] = {name: [] for name in STRATA_ORDER}
    seen_text: set[str] = set()
    seen_emb: list[int] = []

    def compatible(row_idx: int) -> bool:
        if pool.at[row_idx, "normalised"] in seen_text:
            return False
        if seen_emb:
            sims = pool_emb[seen_emb] @ pool_emb[row_idx]
            if float(sims.max()) >= DEDUPE_SIM:
                return False
        return True

    def take(row_idx: int, stratum: str) -> None:
        selected.append(row_idx)
        chosen[stratum].append(row_idx)
        seen_text.add(pool.at[row_idx, "normalised"])
        seen_emb.append(row_idx)

    def stratum_order(stratum: str) -> list[int]:
        """Interleave labels inside intent strata so rare classes are represented."""
        if stratum not in {"rare_intent", "common"}:
            return order
        groups: dict[str, list[int]] = {}
        for row_idx in order:
            if pool.at[row_idx, "stratum"] != stratum:
                continue
            groups.setdefault(str(pool.at[row_idx, "label"]), []).append(row_idx)
        queues = [groups[name] for name in sorted(groups)]
        interleaved: list[int] = []
        while queues:
            for queue in list(queues):
                interleaved.append(queue.pop(0))
                if not queue:
                    queues.remove(queue)
        return interleaved

    for stratum in STRATA_ORDER:
        quota = quotas[stratum]
        for row_idx in stratum_order(stratum):
            if len(chosen[stratum]) >= quota:
                break
            if row_idx in selected:
                continue
            if pool.at[row_idx, "stratum"] != stratum:
                continue
            if compatible(row_idx):
                take(row_idx, stratum)

    # Fill any shortfall from the remaining eligible pool (reported, not hidden).
    shortfall = n - len(selected)
    if shortfall > 0:
        for row_idx in order:
            if len(selected) >= n:
                break
            if row_idx in selected or not compatible(row_idx):
                continue
            take(row_idx, "common")

    if len(selected) < n:
        raise RuntimeError(
            f"only {len(selected)} eligible rows after dedupe; cannot build {n}-query sample"
        )

    sample = pool.loc[selected].copy()
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sample["item_id"] = [f"kcc-{i + 1:04d}" for i in range(len(sample))]

    realised = sample["stratum"].value_counts().to_dict()
    summary = {
        "requested_n": n,
        "realised_n": int(len(sample)),
        "seed": seed,
        "quotas": quotas,
        "realised_strata": {name: int(realised.get(name, 0)) for name in STRATA_ORDER},
        "shortfall_refilled_from_common": int(shortfall),
        "shortfall_by_stratum": {
            name: max(0, quotas[name] - int(realised.get(name, 0))) for name in STRATA_ORDER
        },
        "dedupe": {"text": "normalised exact", "embedding": f"cosine >= {DEDUPE_SIM}"},
        "novelty_thresholds": {"near_duplicate": NEAR_DUPLICATE, "similar": SIMILAR},
    }
    return sample, summary


# --------------------------------------------------------------------------- #
# Item construction
# --------------------------------------------------------------------------- #
def _metadata_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return str(value).strip()


def _truncate(text: str, limit: int = 900) -> tuple[str, bool]:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text, False
    return text[:limit].rstrip() + " …", True


def build_items(sample: pd.DataFrame, pool: pd.DataFrame, seed: int = DEFAULT_SEED) -> tuple[list[dict], list[dict]]:
    """Return (blind items, answer key) for the selected sample.

    ``items`` contains only what an annotator should see.  ``key`` holds the
    stratum, the existing LLM label, the query's source row and the provenance
    of each candidate answer.  The two files must be stored separately.
    """
    rng = random.Random(seed)
    items: list[dict] = []
    key: list[dict] = []

    by_label: dict[str, list[int]] = {}
    for idx, label in pool["label"].items():
        by_label.setdefault(str(label), []).append(idx)

    all_answers = pool["answers"].fillna("").astype(str)
    for _, row in sample.iterrows():
        own_idx = int(row["source_row"])
        own_answer = str(row["answers"] or "").strip()

        candidates: list[dict] = []
        used: set[str] = {own_answer}
        candidates.append({"text": own_answer, "provenance": "logged"})

        # Topic-matched distractor: same existing LLM label, different query.
        same = [i for i in by_label.get(str(row["label"]), []) if i != own_idx]
        rng.shuffle(same)
        for idx in same:
            text = str(all_answers.loc[idx]).strip()
            if len(text) >= 25 and text not in used:
                candidates.append({"text": text, "provenance": "distractor_same_topic"})
                used.add(text)
                break

        # Unrelated distractor: different existing LLM label.
        other_labels = [lab for lab in by_label if lab != str(row["label"])]
        rng.shuffle(other_labels)
        for label in other_labels:
            options = by_label[label][:]
            rng.shuffle(options)
            for idx in options:
                text = str(all_answers.loc[idx]).strip()
                if len(text) >= 25 and text not in used:
                    candidates.append({"text": text, "provenance": "distractor_other_topic"})
                    used.add(text)
                    break
            if len(candidates) == 3:
                break

        if len(candidates) < 3:
            raise RuntimeError(f"could not build three answer candidates for {row['item_id']}")

        rng.shuffle(candidates)
        blind_candidates = []
        provenance = []
        for position, candidate in enumerate(candidates, start=1):
            candidate_id = f"c{position}"
            text, truncated = _truncate(candidate["text"])
            blind_candidates.append({"candidate_id": candidate_id, "text": text})
            provenance.append({
                "candidate_id": candidate_id,
                "provenance": candidate["provenance"],
                "truncated": truncated,
                "original_length": len(candidate["text"]),
            })

        items.append({
            "item_id": row["item_id"],
            "query": str(row["text"]).strip(),
            "candidates": blind_candidates,
            "language": "en",
            "tasks": [
                "intent", "out_of_scope", "multi_intent", "safety",
                "missing_context", "ambiguous", "certainty", "notes",
                "retrieval_relevance", "answer_quality", "answer_safety",
            ],
        })
        key.append({
            "item_id": row["item_id"],
            "source": {
                "dataset": "data/processed/full_labeled.csv",
                "annotation_file": "data/annotation/annotated.csv",
                "annotation_row": own_idx,
                "annotation_id": str(row["id"]),
                "qid": int(row["qid"]),
                "pool": str(row["pool"]),
            },
            "stratum": str(row["stratum"]),
            "llm_label": str(row["label"]),
            "llm_label_provenance": "LLM-generated (Claude), not human-verified",
            "weak_label": str(row["weak_label"]) if "weak_label" in row.index else "",
            "crops": _metadata_text(row.get("crops")),
            "symptoms": _metadata_text(row.get("symptoms")),
            "max_corpus_similarity": round(float(row["max_similarity"]), 6),
            "corpus_group_size": int(row["corpus_group_size"]),
            "near_duplicate_neighbours": int(row["near_duplicate_neighbours"]),
            "candidates": provenance,
        })
    return items, key


def build_sample(n: int = DEFAULT_N, seed: int = DEFAULT_SEED,
                 quotas: dict[str, int] | None = None) -> tuple[list[dict], list[dict], dict]:
    pool, stats = build_pool()
    sample, selection = select_sample(pool, n=n, seed=seed, quotas=quotas)
    items, key = build_items(sample, pool.set_index("source_row"), seed=seed)

    strata = sample["stratum"].value_counts()
    labels = sample["label"].value_counts()
    summary = {
        **stats,
        **selection,
        "eligible_pool_label_counts": pool["label"].value_counts().to_dict(),
        "sample_label_counts": labels.to_dict(),
        "sample_register": {"en": len(sample)},
        "candidate_provenance_counts": {
            prov: sum(1 for item in key for c in item["candidates"] if c["provenance"] == prov)
            for prov in ("logged", "distractor_same_topic", "distractor_other_topic")
        },
        "n_items": len(items),
        "n_candidates": sum(len(item["candidates"]) for item in items),
        "query_length": {
            "min": int(sample["text"].str.len().min()),
            "median": float(sample["text"].str.len().median()),
            "max": int(sample["text"].str.len().max()),
        },
        "items_sha1": _sha1(item["item_id"] + item["query"] for item in items),
        "english_filter": "src/lang.py detect_register == 'en' (read-only import)",
        "sampling_notes": [
            "Strata are assigned by eligibility priority, so a query belongs to exactly one stratum.",
            "Existing LLM labels were used only for stratification and are withheld from annotators.",
            "Queries are real corpus rows; the sample is not synthetic.",
        ],
    }
    return items, key, summary
