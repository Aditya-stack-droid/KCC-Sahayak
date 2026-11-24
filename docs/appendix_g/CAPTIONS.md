# Appendix G figures

- **G01_dataset_preprocessing.png** - Dataset loading and preprocessing output: raw KCC rows, cleaned labelled queries, train/validation/test split and topic distribution.
- **G02_training_results.png** - Model comparison and confidence-gate results on the held-out silver set and the 150-query gold set.
- **G03_voice_input_speech_to_text.png** - Voice input and speech-to-text: audio recorded with the microphone button is transcribed by Whisper and passed through the pipeline.
- **G04_query_classification_pipeline.png** - Farmer query classification: pipeline stages (speech/text, intent with confidence, entities, retrieval, confidence gate) for a typed query.
- **G05_retrieved_historical_qa.png** - Retrieved historical question-and-answer results: closest logged KCC questions with similarity and the advisor answer recorded for each.
- **G06a_safety_gate_auto_answer.png** - Safety gate decision: query answered automatically (non-high-stakes topic, confidence and similarity above threshold, answer screened clean).
- **G06b_safety_gate_escalation.png** - Safety gate decision: high-stakes query (Plant Protection) escalated to a human advisor, with provisional guidance and the closest logged questions.
- **G07_advisor_escalation_console.png** - Human advisor escalation interface: queue of escalated queries, the retrieved candidates and the answer form.
- **G08_homepage.png** - KCC-Sahayak application homepage with the measured evaluation figures.
- **G09a_fastapi_docs.png** - FastAPI backend: auto-generated API documentation listing the query, transcribe, queue and statistics endpoints.
- **G09b_backend_running.png** - FastAPI backend execution: server started with uvicorn and answering the health and model-info endpoints.
