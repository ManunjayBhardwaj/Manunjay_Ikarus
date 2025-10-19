Product Recommendation Web App (FastAPI + React)

Overview
--------
This repository contains a minimal full-stack Product Recommendation Web App for furniture. The backend is built with FastAPI and provides endpoints for health, ingestion, recommendations, and analytics. The frontend is a simple React app that calls the backend APIs.

Tech stack
----------
- Backend: FastAPI, Uvicorn, scikit-learn, sentence-transformers, FAISS (local) or Pinecone (cloud)
- Frontend: React (CRA / Vite), Axios, Recharts
 - ML: sentence-transformers (all-MiniLM-L6-v2), torchvision (optional), LangChain + Gemini/Vertex for GenAI
- Vector DB: Pinecone (preferred) or FAISS fallback

GenAI (Gemini / Vertex AI)
---------------------------
This project supports using Google Gemini via Vertex AI as the preferred LLM. LangChain's
VertexAI wrapper will be used when Vertex credentials are present or when `USE_GEMINI=1` is set.

To use Gemini/Vertex AI:

1. Create a Google Cloud service account with the `Vertex AI User` role and download the JSON key.
2. Set the environment variable in the backend `.env` file:

  GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/service-account.json
  USE_GEMINI=1

3. Install the optional dependency `google-cloud-aiplatform` (already listed in `backend/requirements.txt`).

This project is configured for Gemini-only by default. The priority is:

1. GEMINI_API_KEY (direct REST; set `GEMINI_API_KEY` in `.env`) — used if present.
2. Vertex AI via Google service account (`GOOGLE_APPLICATION_CREDENTIALS`) — used when set or when `USE_GEMINI=1`.
3. Deterministic fallback text when neither is configured.

Local setup
-----------
1. Create a Python venv and install backend requirements:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r backend/requirements.txt

2. Install frontend dependencies and start dev server:

   cd frontend
   npm install
   npm start

Environment variables
---------------------
Create `backend/.env` from `.env.example` and set the following (or leave unset to use FAISS):

- PINECONE_API_KEY=TODO_PINECONE_KEY
- PINECONE_ENV=us-east-1
- PINECONE_INDEX=furniture-index
- VECTOR_BACKEND=faiss
- GEMINI_API_KEY=TODO_GEMINI_KEY  # optional direct key
- GEMINI_API_URL=https://gemini.googleapis.com/v1/models/text-bison:generate  # optional override
- USE_GEMINI=1
- GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/your/service-account.json  # if using Vertex
- HUGGINGFACEHUB_API_TOKEN=
- DATASET_PATH=backend/data/mock_furniture.csv

Dataset
-------
Place your dataset at `backend/data/furniture.csv`. A small mock dataset is included at `backend/data/mock_furniture.csv`.

Running
-------
- Backend (development):

  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

- Frontend (development):

  npm start (in `frontend` folder)

- Ingest (build embeddings and index):

  python backend/scripts/ingest.py --dataset backend/data/mock_furniture.csv --backend faiss

API
---
- GET /api/health — returns {"status":"ok"}
- POST /api/ingest — ingests dataset, builds embeddings, clusters, and upserts to vector DB
- POST /api/recommend — body {query,k} returns top-k products with generated marketing copy
- GET /api/analytics — returns category counts, avg price by category, top brands, material distribution

Deployment
----------
Build the React frontend and serve it from FastAPI in production:

1. cd frontend; npm run build
2. Run the backend (ensure `StaticFiles` in `main.py` points to `../frontend/build`)

For platforms like Render or Railway, build both frontend and backend in a Dockerfile and set env vars there.

Troubleshooting
---------------
- Windows PowerShell: if scripts fail to run, set execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` (run as admin).
- Torch DLL issues on Windows: install a compatible PyTorch wheel from https://pytorch.org and ensure CUDA toolkit is installed if using GPU. On macOS, prefer CPU or use MPS-enabled PyTorch builds.

Notes
-----
This scaffold is intentionally minimal. Use GitHub Copilot to expand each module by pasting the provided prompts into the target files.

---

Quick start
-----------
- From repository root:
  - Backend: cd backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload
  - Frontend: cd frontend && npm install && npm start

If you plan to push to GitHub, ensure large generated folders like `frontend/node_modules` and `frontend/build` are excluded by `.gitignore` (already added).
# Manunjay_Ikarus
