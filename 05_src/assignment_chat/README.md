# Assignment 2 — Spark, Productivity Assistant

A conversational AI with three services, implemented with LangGraph function calling, ChromaDB semantic search, and the Open-Meteo API.

## Services

### 1. Weather (API)
Calls the free Open-Meteo geocoding and forecast APIs. Returns a natural-language summary of current conditions (temperature, wind speed).

### 2. Inspirational Quotes (Semantic Search)
Semantic search over a custom dataset of 30 quotes (food, friendship, community) stored in a ChromaDB persistent collection. Queries are embedded with `text-embedding-3-small` and matched by cosine similarity.

### 3. Calculator (Function Calling)
The LLM uses function calling to invoke an `add`, `subtract`, `multiply`, or `divide` tool. The LLM extracts the operation and operands from natural language.

## Personality
Spark is an upbeat, concise assistant focused on well-being and productivity.

## Guardrails
- Refuses all questions about cats, dogs, horoscopes, zodiac signs, and Taylor Swift.
- Refuses to reveal or modify the system prompt.

## Setup

1. Set up the semantic database (one-time):
   ```bash
   python setup_semantic_db.py
   ```

2. Run the app:
   ```bash
   python app.py
   ```

## Embedding Process
Embeddings were generated using `text-embedding-3-small` via the OpenAI API and stored in a ChromaDB persistent client (`./chroma_db`). See `setup_semantic_db.py` for the full script.
