# Multi-intent safety study (v2, risk-first)

**Evidence class:** authored synthetic cases; rule-based systems; no model training.
The v2 detector is **standalone** and not connected to the production gate.

Risk intents: Government Scheme, Market Price, Nutrient Management, Plant Protection, Weather.

### Original Set (110 cases)

| Metric | baseline (single) | v1 (decompose) | v2 (risk-first) |
|---|---:|---:|---:|
| Exact intent-set match | 0.2182 | 0.7818 | 0.7364 |
| Micro precision | 0.7727 | 0.9278 | 0.9011 |
| Micro recall | 0.4497 | 0.8836 | 0.8677 |
| Micro F1 | 0.5686 | 0.9051 | 0.8841 |
| Macro F1 | 0.6288 | 0.9131 | 0.8907 |
| Missed-intent rate | 0.5503 | 0.1164 | 0.1323 |
| Extra-intent rate | 0.2273 | 0.0722 | 0.0989 |
| Decomposition accuracy | 0.0000 | 0.9054 | 0.8514 |
| High-risk intent recall | 0.7375 | 0.9500 | 0.9500 |
| Safety escalation precision | 1.0000 | 1.0000 | 0.9195 |
| Safety escalation recall | 0.7375 | 0.9500 | 1.0000 |
| **High-risk false negatives** | 21 | 4 | **0** |

### Adversarial Set (110 cases)

| Metric | baseline (single) | v1 (decompose) | v2 (risk-first) |
|---|---:|---:|---:|
| Exact intent-set match | 0.1545 | 0.7091 | 0.6091 |
| Micro precision | 0.6364 | 0.8757 | 0.8516 |
| Micro recall | 0.3646 | 0.8438 | 0.8073 |
| Micro F1 | 0.4636 | 0.8594 | 0.8289 |
| Macro F1 | 0.6121 | 0.9398 | 0.8993 |
| Missed-intent rate | 0.6354 | 0.1562 | 0.1927 |
| Extra-intent rate | 0.3636 | 0.1243 | 0.1484 |
| Decomposition accuracy | 0.0000 | 0.8987 | 0.7848 |
| High-risk intent recall | 0.5773 | 0.8763 | 0.9072 |
| Safety escalation precision | 1.0000 | 1.0000 | 1.0000 |
| Safety escalation recall | 0.5773 | 0.8763 | 1.0000 |
| **High-risk false negatives** | 41 | 12 | **0** |

### Combined (220 cases)

| Metric | baseline (single) | v1 (decompose) | v2 (risk-first) |
|---|---:|---:|---:|
| Exact intent-set match | 0.1864 | 0.7455 | 0.6727 |
| Micro precision | 0.7045 | 0.9014 | 0.8764 |
| Micro recall | 0.4068 | 0.8635 | 0.8373 |
| Micro F1 | 0.5158 | 0.8820 | 0.8564 |
| Macro F1 | 0.5866 | 0.8877 | 0.8558 |
| Missed-intent rate | 0.5932 | 0.1365 | 0.1627 |
| Extra-intent rate | 0.2955 | 0.0986 | 0.1236 |
| Decomposition accuracy | 0.0000 | 0.9020 | 0.8170 |
| High-risk intent recall | 0.6497 | 0.9096 | 0.9266 |
| Safety escalation precision | 1.0000 | 1.0000 | 0.9620 |
| Safety escalation recall | 0.6497 | 0.9096 | 1.0000 |
| **High-risk false negatives** | 62 | 16 | **0** |

## Limitations

- Cases are synthetic and authored by the module's author; not real traffic.
- gold_safety is derived from authored gold intents; a mislabelled gold intent would mislabel safety.
- v2 escalates more by design (conservative); this lowers escalation precision on ambiguous cases.
- Rule-based confidence is a heuristic, not a calibrated probability.
