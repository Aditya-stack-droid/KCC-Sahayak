"""Build the authored synthetic pair set for conflict detection.

Every row is synthetic and hand-labelled for this evaluation.  It is not a
sample of real advisor disagreement and is not expert-verified.  The set mixes
conflict types with near-miss negatives (formatting changes, 10–20% dose
differences, different crops, pack sizes, prices, negated pairs).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "conflict_pairs.csv"
ALLOWED = {"dosage", "product", "basis", "dimension", "interval", "polarity"}

PAIRS: list[tuple[str, str, str, str]] = [
    # ---- dosage differences ------------------------------------------------
    ("d01", "Apply urea 50 kg per acre at sowing.", "Apply urea 20 kg per acre at sowing.", "dosage"),
    ("d02", "Spray 2 ml chlorpyrifos per litre of water.", "Spray 5 ml chlorpyrifos per litre of water.", "dosage"),
    ("d03", "Mix 20 gm mancozeb in 10 litre water.", "Mix 40 gm mancozeb in 10 litre water.", "dosage"),
    ("d04", "Apply 1 kg zinc sulphate per bigha.", "Apply 4 kg zinc sulphate per bigha.", "dosage"),
    ("d05", "Use 500 ml pendimethalin per acre.", "Use 1 litre pendimethalin per acre.", "dosage"),
    ("d06", "Apply 2 quintal FYM per acre.", "Apply 50 kg FYM per acre.", "dosage"),
    ("d07", "Dilute 1 ml imidacloprid in 1 litre water.", "Dilute 1 ml imidacloprid in 2 litre water.", "dosage"),
    ("d08", "Apply 10 kg DAP per acre.", "Apply 25 kg DAP per acre.", "dosage"),
    ("d09", "Apply 2 g carbendazim per litre.", "Apply 1 g carbendazim per litre.", "dosage"),
    ("d10", "Apply urea 10 kg per acre.", "Apply urea 12 kg per acre.", ""),
    ("d11", "Apply urea 2-3 kg per acre.", "Apply urea 3 kg per acre.", ""),
    ("d12", "Apply 500 gm urea per plant.", "Apply 0.5 kg urea per plant.", ""),
    ("d13", "Apply 2% urea spray.", "Apply 2 percent urea solution.", ""),
    ("d14", "Apply 1 kg urea per 2 acres.", "Apply 0.5 kg urea per acre.", ""),
    ("d15", "Apply 1 kg urea per 2 acres.", "Apply 1 kg urea per acre.", "dosage"),
    ("d16", "Apply 40 kg urea per acre.", "Apply 20 kg urea per 2 acres.", "dosage"),
    ("d17", "Apply 5 kg urea.", "Apply 5 litre urea.", "dimension"),
    ("d18", "Spray 10 gm sulphur per litre.", "Spray 10 ml sulphur per litre.", "dimension"),
    ("d19", "Spray 2 ml dichlorvos per litre.", "Spray 8 ml dichlorvos per litre.", "dosage"),
    ("d20", "Apply 4 kg zinc sulphate per acre.", "Apply 8 kg zinc sulphate per acre.", "dosage"),
    # ---- different products ------------------------------------------------
    ("p01", "Spray rogor for aphid control.", "Spray mancozeb for aphid control.", "product"),
    ("p02", "Apply urea at sowing.", "Apply DAP at sowing.", "product"),
    ("p03", "Use chlorpyrifos 2 ml per litre.", "Use imidacloprid 2 ml per litre.", "product"),
    ("p04", "Spray neem oil in the evening.", "Spray malathion in the evening.", "product"),
    ("p05", "Apply captan 3 g per litre.", "Apply carbendazim 3 g per litre.", "product"),
    ("p06", "Use glyphosate before sowing.", "Use pendimethalin before sowing.", "product"),
    ("p07", "Spray copper oxychloride.", "Spray bordeaux mixture.", "product"),
    ("p08", "Apply MOP at flowering.", "Apply potash at flowering.", ""),
    ("p09", "Apply urea or DAP.", "Apply urea.", ""),
    ("p10", "Add vermicompost to the nursery.", "Add FYM to the nursery.", "product"),
    ("p11", "Apply urea.", "Contact the agriculture officer for details.", ""),
    ("p12", "Spray neem oil.", "Spray neem kernel extract.", ""),
    ("p13", "Apply 50 kg urea per acre.", "Apply 50 kg DAP per acre.", "product"),
    ("p14", "Spray rogor 2 ml per litre.", "Spray mancozeb 5 gm per litre.", "product"),
    ("p15", "Apply ammonium sulphate.", "Apply urea.", "product"),
    # ---- basis / unit ------------------------------------------------------
    ("b01", "Apply 50 kg urea per acre.", "Apply 50 kg urea per bigha.", "basis"),
    ("b02", "Apply 50 kg urea per hectare.", "Apply 50 kg urea per acre.", "dosage"),
    ("b03", "Apply 2 kg urea per plant.", "Apply 2 kg urea per acre.", "basis"),
    ("b04", "Apply 2 kg zinc sulphate per bigha.", "Apply 2 kg zinc sulphate per acre.", "basis"),
    ("b05", "Apply 1 kg urea per kanal.", "Apply 1 kg urea per hectare.", "basis"),
    ("b06", "Spray 2 ml chlorpyrifos per litre.", "Spray 2 ml chlorpyrifos per 10 litre.", "dosage"),
    ("b07", "Spray 20 gm copper oxychloride per litre.", "Spray 20 ml copper oxychloride per litre.", "dimension"),
    ("b08", "Apply 10 kg DAP.", "Apply 10 litre DAP.", "dimension"),
    # ---- intervals ---------------------------------------------------------
    ("i01", "Spray after 5 days.", "Spray after 15 days.", "interval"),
    ("i02", "Repeat after 10 days.", "Repeat after 12 days.", ""),
    ("i03", "Spray at an interval of 5 days.", "Spray at an interval of 15 days.", "interval"),
    ("i04", "Apply again after 7 days.", "Apply again after 7 days.", ""),
    ("i05", "Apply again after 5 days.", "Apply again after 20 days.", "interval"),
    ("i06", "Spray at 5 day interval.", "Spray at 5 day interval.", ""),
    # ---- negation / polarity ----------------------------------------------
    ("x01", "Do not spray chlorpyrifos.", "Spray chlorpyrifos 2 ml per litre.", "polarity"),
    ("x02", "Avoid urea at sowing.", "Apply urea at sowing.", "polarity"),
    ("x03", "Do not spray chlorpyrifos.", "Do not apply chlorpyrifos.", ""),
    ("x04", "Never use glyphosate.", "Glyphosate is not recommended.", ""),
    ("x05", "Do not apply urea; instead spray neem oil.", "Spray neem oil.", ""),
    ("x06", "Do not use DAP.", "Use urea.", "product"),
    ("x07", "Do not apply urea.", "Apply 20 kg urea per acre.", "polarity"),
    ("x08", "Avoid chlorpyrifos.", "Chlorpyrifos 2 ml per litre is effective.", "polarity"),
    # ---- negative controls -------------------------------------------------
    ("c01", "Apply urea 50 kg per acre.", "Sell wheat at the mandi.", ""),
    ("c02", "Apply urea 50 kg per acre at sowing.", "Apply 50 kg urea per acre.", ""),
    ("c03", "Apply urea 50 kg per acre in wheat.", "Apply urea 50 kg per acre in paddy.", ""),
    ("c04", "Available in 500 gm pack.", "Apply 500 gm urea per acre.", ""),
    ("c05", "Urea price is Rs 266 per bag.", "Apply 50 kg urea per acre.", ""),
    ("c06", "Contact the KVK for soil testing.", "Visit the district agriculture office.", ""),
    ("c07", "Apply 2 kg urea per acre.", "Apply 2 kg urea per bigha.", "basis"),
    ("c08", "Spray 2 ml per litre in the evening.", "Spray 2 ml per litre.", ""),
    ("c09", "Apply 50 kg urea per acre.", "Apply 50 kg urea per acre.", ""),
    ("c10", "Mix 2 ml neem oil in 1 litre water.", "Mix 2 ml neem oil in 1 litre water at dusk.", ""),
]


# Cases written after the core rules existed, to expose likely failure modes
# (obfuscated products, non-standard units, Hindi local area units, action-level
# negation).  They are reported separately from the core set; this is not a
# held-out benchmark either, because the author knew the rules when writing them.
CHALLENGE: list[tuple[str, str, str, str]] = [
    ("h01", "Spray r0gor for aphid control.", "Spray mancozeb for aphid control.", "product"),
    ("h02", "Apply 2 sachet urea per acre.", "Apply 5 sachet urea per acre.", "dosage"),
    ("h03", "Apply 1 bag urea per acre.", "Apply 3 bag urea per acre.", "dosage"),
    ("h04", "Do not spray any chemical.", "Spray chlorpyrifos 2 ml per litre.", "polarity"),
    ("h05", "यूरिया 50 किलो प्रति एकड़ डालें।", "यूरिया 50 किलो प्रति बीघा डालें।", "basis"),
    ("h06", "यूरिया २ किलो प्रति एकड़ डालें।", "यूरिया ५ किलो प्रति एकड़ डालें।", "dosage"),
    ("h07", "Apply ur3a 50 kg per acre.", "Apply urea 20 kg per acre.", "dosage"),
    ("h08", "Spray chlorpyriphos 2 ml per litre.", "Spray chlorpyrifos 5 ml per litre.", "dosage"),
    ("h09", "Apply 2 kg urea per acre.", "Apply 2 kg urea per acre.", ""),
    ("h10", "Spray 2 ml per litre of the same insecticide.", "Spray 5 ml per litre of the same insecticide.", "dosage"),
]


def rows() -> list[dict]:
    output = []
    for subset, entries in (("core", PAIRS), ("challenge", CHALLENGE)):
        for pair_id, text_a, text_b, kinds in entries:
            labels = [kind for kind in kinds.split(";") if kind]
            unknown = set(labels) - ALLOWED
            if unknown:
                raise ValueError(f"{pair_id}: unknown conflict kind(s) {unknown}")
            output.append({
                "pair_id": pair_id,
                "text_a": text_a,
                "text_b": text_b,
                "gold_conflict": "true" if labels else "false",
                "gold_kinds": ";".join(labels),
                "subset": subset,
                "synthetic": "true",
                "provenance": "hand-authored for the Batch B conflict-detection evaluation; not real advisor data",
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.overwrite:
        print(f"refusing to overwrite {OUT} (pass --overwrite)")
        return 1
    DATA.mkdir(parents=True, exist_ok=True)
    data = rows()
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    positive = sum(1 for row in data if row["gold_conflict"] == "true")
    print(f"wrote {OUT}: {len(data)} pairs ({positive} conflict, {len(data) - positive} no-conflict)")
    counts: dict[str, int] = {}
    for row in data:
        for kind in row["gold_kinds"].split(";"):
            if kind:
                counts[kind] = counts.get(kind, 0) + 1
    print("kinds:", counts)
    print("subsets:", {subset: sum(1 for row in data if row["subset"] == subset)
                       for subset in ("core", "challenge")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
