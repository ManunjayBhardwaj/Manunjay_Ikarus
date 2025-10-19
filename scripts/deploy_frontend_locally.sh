#!/usr/bin/env bash
# convenience script: build the frontend from repo root (same as Netlify netlify.toml)
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
cd frontend
echo "Installing frontend deps and building..."
npm install
npm run build
echo "Build complete: frontend/build"
