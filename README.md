# Nexus: Centralized Retail Intelligence Platform

## Overview
A prototype platform to unify intelligence across inventory, logistics, returns, and supplier relations for large-scale retailers. Enables natural language queries and predictive insights using graph databases, vector search, and LLMs.

## Architecture
- **Data Ingestion Layer:** Mock CSV data for inventory, suppliers, logistics, returns
- **Graph DB Layer:** Neo4j for relationship modeling
- **Vector Store Layer:** Sentence Transformers + FAISS
- **RAG Layer:** Retrieval-Augmented Generation pipeline with OpenAI LLM
- **Frontend:** Streamlit app for NL queries and visualization

## Setup
1. Clone repo and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your API keys and DB URIs to `.env`.
3. Run the Streamlit app:
   ```bash
   streamlit run frontend/app.py
   ```

## Folder Structure
- `data/` - Mock data CSVs
- `vector_db/` - Embedding/vector code
- `graph_db/` - Neo4j code
- `rag/` - RAG pipeline
- `frontend/` - Streamlit app 