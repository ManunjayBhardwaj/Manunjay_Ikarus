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
# Ikarus — Furniture Recommendation Web App

Small demo app (FastAPI + React) that demonstrates ingestion, simple ML embeddings, a recommender, analytics, and a small GenAI-backed chat assistant. This repository was prepared as an internship / demo project and includes data-cleaning and training utilities.

Status summary
- Backend: FastAPI app in `backend/app/` with endpoints for recommend, analytics, ingest, and chat. Run with uvicorn.
- Frontend: React (Create React App) in `frontend/` with pages for Recommend, Analytics, and Chat.
- ML utilities: scripts in `backend/scripts/` to compute embeddings, cluster, and export artifacts to `backend/backend/storage/`.
- Data: original and cleaned data under `backend/data/`.

Table of contents
- Quick start
- Backend (detailed)
- Frontend (detailed)
- ML & data scripts
- GenAI / model notes
- Project layout
- Troubleshooting
- License

Quick start (dev)

Prerequisites
- Python 3.10+ (3.11/3.12 tested). A virtualenv or conda env is recommended.
- Node 16+ / npm 8+ for the frontend.
- If you plan to use GenAI (Gemini / Vertex) ensure you have credentials and network access.

1) Backend (API)

Install deps and run the API (from repo root):

```bash
cd backend
python -m venv .venv            # optional
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
# start uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir app
```

Health check: http://127.0.0.1:8000/api/health

Common backend files
- `backend/app/main.py` — FastAPI app wiring and router includes.
- `backend/app/routes/` — handlers for `/recommend`, `/analytics`, `/chat`, `/ingest`.
- `backend/app/models/genai.py` — wrapper to call Gemini/Vertex and deterministic fallbacks.
- `backend/backend/storage/` — generated artifacts (embeddings.npy, metadata.json, training_report.json) after running training.

2) Frontend (dev)

Install and run the React app:

```bash
cd frontend
npm install
npm start
```

Open http://localhost:3000

3) Data cleaning & analytics

The repository contains a cleaning script used to produce `backend/data/cleaned_furniture.csv` and `analytics.json`.

```bash
cd backend/scripts
python clean_dataset.py --input ../../intern_data_ikarus.csv --outdir ../data --parquet
```

Outputs:
- `backend/data/cleaned_furniture.csv`
- `backend/data/analytics.json` (consumed by `/api/analytics`)

4) Training embeddings & clusters (vector data)

Compute embeddings and clustering (will produce artifacts under `backend/backend/storage`):

```bash
cd backend/scripts
python train_models.py --input ../data/cleaned_furniture.csv --outdir ../backend/storage --model all-MiniLM-L6-v2
```

Notes: FAISS may fail to import on some systems (native build/ABI). The training script falls back to saving numpy embeddings and the app will perform in-process similarity search when FAISS is unavailable.

GenAI / chat assistant
- `backend/app/models/genai.py` implements multi-path GenAI: attempt Gemini REST when `GEMINI_API_KEY` is present, fallback to Vertex via LangChain if configured, and finally a deterministic templated generator (temperature-aware) if external APIs are unavailable.
- To enable Gemini: export `GEMINI_API_KEY` into the environment before starting uvicorn. Logs are written to `/tmp/genai_debug.log`.

Project layout (important files)

```
backend/
  app/
    main.py
    routes/
    models/
  scripts/
    clean_dataset.py
    train_models.py
  data/
    cleaned_furniture.csv
    analytics.json
frontend/
  src/
    pages/
    components/
    styles/
  package.json
notebooks/
  Data_Analytics.ipynb
  Model_Training.ipynb
```

Troubleshooting
- If the frontend shows a blank page after edits, run `npm run build` to regenerate the build and restart the dev server.
- If FAISS import fails, the recommender falls back to numpy similarity. For production, install `faiss-cpu` from a compatible wheel or use a containerized image.
- GenAI: ensure `GEMINI_API_KEY` is exported for Gemini REST calls. If you don't have access, the fallback templated responses will be used.

Contributing & next steps
- Add `gh` CLI and run `gh repo create Manunjay_102203009_Ikarus --public --source=. --remote=origin --push` to create the GitHub repo and push (requires you to be logged in with `gh auth login`).
- Or create a new repo on GitHub web UI and add the remote:

```bash
git remote add origin https://github.com/<your-username>/Manunjay_102203009_Ikarus.git
git branch -M main
git push -u origin main
```

License
MIT — see `LICENSE` file (if present). This project template uses permissive license by default.

Contact
If you need help, open an issue or contact the project owner: Manunjay Bhardwaj

