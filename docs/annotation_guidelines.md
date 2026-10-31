# Annotation guidelines (QueryType)

Labels describe **what the farmer is asking for**, judged from the meaning of
the query, never from keyword presence. One label per query: the primary need.

| Code | QueryType | Use when the farmer wants… |
|---|---|---|
| PP | Plant Protection | to identify or control a pest, disease, weed, or unexplained crop/orchard disorder (wilting, rotting, spots, bleeding, fruit/flower drop with a disease-like cause); includes chemical, biological and IPM control |
| NM | Nutrient Management | fertiliser / manure / micronutrient advice: doses, schedules, deficiency correction, biofertiliser, soil fertility |
| WE | Weather | forecasts, rainfall, frost, hail, heat, monsoon timing, weather alerts |
| GS | Government Scheme | schemes, subsidies, loans and Kisan Credit Card, crop insurance, PM-Kisan, registration, compensation, banking procedure |
| MP | Market Price | mandi rates, MSP rates, where/when to sell, price trends, buyers, marketing |
| VI | Varietal Information | which variety / hybrid to grow, seed sources and availability, variety suitability |
| AH | Animal Husbandry | livestock, dairy, poultry, goat/sheep health, feed and fodder for animals, breeding, vaccination, bee-keeping |
| FI | Fisheries | fish and prawn farming, ponds, fish feed, fish disease, hatchery |
| CP | Cultural Practices | agronomy: sowing time, seed rate, spacing, irrigation scheduling, pruning/training, intercropping, harvest timing, land preparation, nursery, rotation |
| OT | Other | anything else: soil/water testing, storage and post-harvest handling, processing, farm machinery use, training programmes, contact details, mushroom cultivation, general information, unintelligible or empty queries |

## Tie-breaks

- Pest/disease control **and** a dose → PP (the problem comes first).
- Pure fertiliser dose with no pest/disease → NM.
- A subsidy/loan **for** a machine or irrigation system → GS (not OT).
- "Price of X seed" → VI if about where to get/which variety, MP if about the rate.
- Fodder *crop cultivation for cattle* → AH; fodder agronomy such as sowing time → CP.
- When the text is too short or garbled to tell, → OT.

## Provenance (disclosed in the report)

The gold set (`data/gold/`) and the training/silver annotations
(`data/annotation/`) were produced by an LLM assistant (Claude) following
these rules, without human verification. They share one annotator, so
measured agreement between them includes shared annotator bias. A
`human_label` column is provided in `data/gold/gold_queries_review.csv` so a
person can verify a sample and compute agreement (`scripts/10_agreement.py`).
