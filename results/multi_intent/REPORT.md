# Multi-intent detection: single-label baseline vs rule-based decomposition

**Evidence class:** authored synthetic cases; rule-vs-rule comparison; no model loaded,
no human labels and no expert review. The decomposer is **standalone and escalate-only**
and is not wired into the production gate.

**Cases:** 110 — 74 multi-intent, 36 single-request controls, 80 with a high-stakes/time-sensitive intent.

## 1. Headline comparison

| Metric | single-label baseline | rule-based decomposition |
|---|---:|---:|
| Exact intent-set match | 0.2182 | 0.7818 |
| Micro precision | 0.7727 | 0.9278 |
| Micro recall | 0.4497 | 0.8836 |
| Micro F1 | 0.5686 | 0.9051 |
| Macro F1 | 0.6289 | 0.9130 |
| Missed-intent rate | 0.5503 | 0.1164 |
| Extra-intent rate | 0.2273 | 0.0722 |
| Decomposition accuracy | 0.0000 | 0.9054 |
| Safety-escalation recall | 0.7375 | 0.9500 |
| Safety-escalation precision | 1.0000 | 1.0000 |

## 2. By language

| Language | n | baseline micro-F1 | decomposition micro-F1 | decomposition exact |
|---|---:|---:|---:|---:|
| devanagari | 10 | n/a | 0.9697 | 0.9000 |
| en | 81 | 0.6727 | 0.8856 | 0.7407 |
| hinglish | 8 | 0.5455 | 1.0000 | 1.0000 |
| roman_hindi | 11 | 0.3333 | 0.9189 | 0.8182 |

## 3. Residual failures (not hidden)

- Cases where the decomposer **missed** a gold intent: 21
- Cases where the decomposer **added** an intent: 13
- High-risk queries not flagged for escalation: ['mi-036', 'mi-067', 'mi-087', 'mi-109']

Missed-intent examples:

| Case | Gold | Predicted | Missing |
|---|---|---|---|
| mi-001 | Market Price; Nutrient Management | Nutrient Management; Other | Market Price |
| mi-020 | Cultural Practices; Plant Protection | Other; Plant Protection | Cultural Practices |
| mi-026 | Market Price; Nutrient Management; Varietal Information | Nutrient Management; Varietal Information | Market Price |
| mi-027 | Cultural Practices; Market Price; Plant Protection | Cultural Practices; Market Price | Plant Protection |
| mi-028 | Animal Husbandry; Fisheries; Weather | Animal Husbandry; Weather | Fisheries |
| mi-029 | Nutrient Management; Plant Protection; Varietal Information | Nutrient Management; Plant Protection | Varietal Information |
| mi-030 | Market Price; Nutrient Management; Varietal Information | Market Price; Nutrient Management | Varietal Information |
| mi-036 | Plant Protection; Varietal Information | Varietal Information | Plant Protection |
| mi-040 | Nutrient Management; Plant Protection | Nutrient Management | Plant Protection |
| mi-057 | Cultural Practices; Plant Protection | Cultural Practices; Nutrient Management | Plant Protection |
| mi-067 | Plant Protection | Other | Plant Protection |
| mi-081 | Cultural Practices | Other | Cultural Practices |
| mi-086 | Market Price; Nutrient Management | Nutrient Management | Market Price |
| mi-087 | Cultural Practices; Market Price | Other | Cultural Practices; Market Price |
| mi-088 | Cultural Practices; Plant Protection | Plant Protection | Cultural Practices |
| mi-091 | Cultural Practices; Nutrient Management | Nutrient Management; Other | Cultural Practices |
| mi-096 | Market Price; Nutrient Management | Nutrient Management; Other | Market Price |
| mi-097 | Cultural Practices; Varietal Information | Varietal Information | Cultural Practices |
| mi-108 | Cultural Practices; Varietal Information | Other; Varietal Information | Cultural Practices |
| mi-109 | Animal Husbandry; Market Price | Animal Husbandry | Market Price |

## 4. Limitations

- Cases are synthetic and authored by the decomposer's author; not real traffic.
- Gold intent sets are authored judgements; multi-intent labelling is inherently subjective.
- The baseline is the keyword rule, not the deployed ML classifier (no model loaded here).
- The decomposer reuses English keyword rules plus a small bilingual cue lexicon; it is not a trained multi-label model.
- It is not wired into the production gate; it would only add an escalation signal.
