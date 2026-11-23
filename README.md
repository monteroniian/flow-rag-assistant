# Flow · Agente RAG de Psicología

Asistente conversacional local basado en **RAG (Retrieval-Augmented Generation)** con **FastAPI**, **LangChain**, **ChromaDB**, **Ollama** (
omic-embed-text + phi3:3.8b) y síntesis de voz **Kokoro TTS**.

## Características
- Base vectorial local con ChromaDB e ingesta documental (ingestar_documentos.py)
- Agente supervisor de seguridad con phi3:3.8b
- API REST en FastAPI (pi.py) con respuestas de audio WAV mediante Kokoro TTS
- Interfaz web conectada (Fronted.html)
