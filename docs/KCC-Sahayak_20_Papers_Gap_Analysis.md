# KCC-Sahayak — 20-Paper Literature Survey with Gap Analysis

Ordered most → least important to your project's core claim.

---

### 1. Rehman, M. Z. U., Raghuvanshi, D., & Kumar, N. (2023). *KisanQRS: A deep learning-based automated query-response system for agricultural decision-making.* Computers and Electronics in Agriculture, 213, 108180.
**Gap:** Works only on already-transcribed KCC text; uses LSTM + threshold clustering, not a fine-tuned transformer; has no explicit human-escalation mechanism — a query is always answered, never deferred.
**What KCC-Sahayak tackles:** Adds the voice-to-text front end (raw calls, not clean text) and replaces "always answer" with a confidence-gated decision to auto-answer or escalate.

### 2. Didwania, K., Seth, P., Kasliwal, A., & Agarwal, A. (2024). *AgriLLM: Harnessing Transformers for Farmer Queries.* arXiv:2407.04721.
**Gap:** Uses a generative LLM to produce fresh answers — no grounding guarantee, so it can hallucinate; no safety layer to withhold an answer on high-stakes queries (e.g., pesticide dosage).
**What KCC-Sahayak tackles:** Uses retrieval over real, historically verified KCC answers instead of open generation, and explicitly withholds/escalates high-stakes or low-confidence cases rather than generating a plausible-sounding but unverified answer.

### 3. *KrushiMitra — Expert System for Farmers using Kisan Call Center Data.* JETIR.
**Gap:** Classical ML (decision trees) with no semantic understanding of free text — cannot handle paraphrased or noisy spoken-style queries well, and predates transformer-based representation.
**What KCC-Sahayak tackles:** Uses fine-tuned Indic transformer embeddings for semantic matching, robust to paraphrasing and informal spoken phrasing that keyword/rule-based classifiers miss.

### 4. Kumar, R. et al. (2024). *Uncovering Farmers' Information Need through Kisan Call Centre Data Analytics of Haryana State.* Indian Journal of Extension Education, 60(4), 59–66.
**Gap:** Purely descriptive data analytics (what farmers ask, when) — no system is built; the paper stops at insight, not action.
**What KCC-Sahayak tackles:** Operationalizes the same Haryana dataset into a working triage system rather than a static analysis, turning the identified query patterns into training data for classification/retrieval.

### 5. *Assessment of Queries of Farmers at Kisan Call Center using NLP.* ResearchGate, 2021.
**Gap:** Focuses only on NER/entity-extraction and semi-automatic annotation; does not connect entity extraction to a downstream retrieval or escalation decision.
**What KCC-Sahayak tackles:** Uses the same bootstrapped-glossary NER approach but wires its output (crop, symptom, pest) directly into the retrieval and confidence-gating pipeline, rather than treating extraction as an end in itself.

### 6. Pramanik, C. *Access to Digital Extension Services and Crop Yield: Evidence From Paddy Cultivation in India.* Agricultural Economics (Tata-Cornell Institute study).
**Gap:** Econometric/outcomes study — shows call *content* correlates with yield outcomes, but offers no mechanism to act on that signal in real time.
**What KCC-Sahayak tackles:** The confidence-gate + district-aggregation stretch goal operationalizes exactly this insight — turning query-topic signal into a live pest/weather-spike detector rather than a retrospective correlation.

### 7. Sahu, S., Bishnoi, S., Sharma, P. R., et al. (2024). *Exploring popular information sources and determinants of farmers' access to agricultural extension services in the Indo-Gangetic plains.* Frontiers in Sustainable Food Systems, 8, 1339243.
**Gap:** Documents *which* channels farmers use and *why* access varies, but is source-agnostic — doesn't address how any one channel (like KCC) could be made more scalable.
**What KCC-Sahayak tackles:** Targets the specific high-access-but-bottlenecked channel (KCC in Haryana) this paper identifies, directly improving throughput on the channel farmers already prefer.

### 8. Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
**Gap:** General-purpose ASR, not tuned or evaluated on Indian regional/rural telephone audio, code-switched Hindi-English, or Haryanvi dialect variation.
**What KCC-Sahayak tackles:** Uses Whisper as the ASR front end but treats its output as noisy and applies text normalization specifically for Hindi-English code-switching before downstream NLP — acknowledging rather than assuming ASR is dialect-robust.

### 9. Khanuja, S. et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730.
**Gap:** A general-purpose Indic language model, not fine-tuned for any specific downstream task like agricultural intent classification.
**What KCC-Sahayak tackles:** Fine-tunes MuRIL specifically on labeled Haryana KCC QueryType data, adapting a general Indic encoder to the agricultural-helpline domain.

### 10. Kakwani, D. et al. (2020). *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages* (IndicBERT), AI4Bharat.
**Gap:** Same as above — general-purpose Indic pretraining, evaluated on generic NLP benchmarks, not on noisy spoken agricultural queries.
**What KCC-Sahayak tackles:** Provides a second candidate backbone to benchmark against MuRIL specifically on the KCC intent-classification task, rather than assuming one generic model is best.

### 11. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.*
**Gap:** Designed and evaluated on general semantic similarity/STS benchmarks (mostly English), not on Indian-language or code-mixed agricultural text.
**What KCC-Sahayak tackles:** Applies the sentence-embedding approach to multilingual Hindi/Haryanvi farmer queries for retrieval, testing whether the method generalizes to this domain and language setting.

### 12. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs* (FAISS).
**Gap:** A general-purpose ANN search library — no domain-specific indexing strategy for structured agricultural metadata (crop, sector, district).
**What KCC-Sahayak tackles:** Builds a FAISS index specifically over Haryana Agriculture/Horticulture historical query-answer pairs, with retrieval scoped by the extracted entities rather than raw text similarity alone.

### 13. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (RAG).
**Gap:** General RAG assumes retrieved context is used to generate an answer; it doesn't address a scenario where the system should sometimes refuse to answer at all.
**What KCC-Sahayak tackles:** Adapts the retrieve-then-respond pattern but adds a hard decision boundary (the confidence gate) that RAG itself doesn't formalize — retrieval quality directly determines whether to answer or escalate, not just what to generate.

### 14. Geifman, Y. & El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks.*
**Gap:** Formalizes the risk-coverage trade-off for abstention in general deep classifiers; not applied to any human-helpline or agricultural-advisory setting.
**What KCC-Sahayak tackles:** Applies selective-classification theory concretely to the KCC use case — defining the confidence threshold as a tunable risk-coverage decision between automation rate and answer safety.

### 15. Kamath, A., Jia, R., & Liang, P. (2020). *Selective Question Answering under Domain Shift.*
**Gap:** Studies when a QA system should abstain under distribution shift, but in a general open-domain QA setting, not one with a human fallback channel already in place (like KCC's live FTAs).
**What KCC-Sahayak tackles:** Extends the "abstain under uncertainty" idea into a real, already-existing human-in-the-loop system — low-confidence cases don't just get refused, they get routed to a human FTA, closing the loop that pure abstention leaves open.

### 16. Ramesh, G. et al. (2022). *Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages* (AI4Bharat).
**Gap:** Provides parallel corpora/translation models for standard Indic languages; doesn't include Haryanvi-specific or rural-telephone-register text.
**What KCC-Sahayak tackles:** Uses IndicTrans2 (built on this data) for normalization but supplements it with domain-specific handling for Haryanvi and rural spoken registers not well represented in Samanantar's source corpora.

### 17. Khanuja, S. et al. (2020). *GLUECoS: An Evaluation Benchmark for Code-Switched NLP.*
**Gap:** Benchmarks code-switching (Hindi-English) NLP tasks like sentiment and NER, but not in an agricultural or voice-transcribed domain.
**What KCC-Sahayak tackles:** Applies code-switching-aware normalization to real Hindi-English mixed farmer speech transcripts, a domain GLUECoS doesn't cover.

### 18. Ajawan, P., Desai, P., & Desai, V. (2020). *Smart Sampark — An approach towards building a responsive system for Kisan Call Center.* IEEE B-HTC.
**Gap:** An early, pre-deep-learning attempt at making KCC more responsive; lacks modern NLP techniques (transformers, semantic retrieval) and lacks any escalation/safety design.
**What KCC-Sahayak tackles:** Picks up the same responsiveness goal with modern transformer-based methods and adds the safety-first escalation layer this earlier system never had.

### 19. Kavitha, S. & Anandaraja, N. (2018). *Kisan Call Centre Services to the Farming Community: An Analysis.*
**Gap:** Survey-based study of farmer-perceived benefits of KCC (cost savings, pest forecasting); doesn't examine or propose any technical improvement to the service itself.
**What KCC-Sahayak tackles:** Targets the exact benefits farmers report valuing (fast pest/disease guidance) and tries to make that specific value available instantly for routine cases.

### 20. Chow, C. K. (1970). *On Optimum Recognition Error and Reject Tradeoff.*
**Gap:** The foundational, decades-old theoretical treatment of error-vs-reject trade-offs in pattern recognition — general and abstract, with no connection to modern deep learning, NLP, or any real-world deployed helpline.
**What KCC-Sahayak tackles:** Grounds the project's confidence-gate design in this formal reject-option theory while applying it concretely, for the first time in this literature, to a live agricultural voice-helpline triage setting.

---

*Note: for papers 8–17 and 20, cite the original methodology papers as tool/technique references (they justify **why** you chose that component), not as prior systems solving your exact problem — keep that distinction clear if a mentor asks "so what's new."*
