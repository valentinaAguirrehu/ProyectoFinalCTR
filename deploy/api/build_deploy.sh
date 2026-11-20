#!/usr/bin/env bash
# Arma en build/api-deploy/ el repositorio que se despliega en Render.
# Conserva su carpeta .git para poder hacer commit y push de cada actualización.
# Requiere los modelos entrenados: python -m src.ml.train
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/build/api-deploy"

[ -f "$ROOT/models/catboost/metadata.json" ] || { echo "Falta models/. Ejecuta: python -m src.ml.train" >&2; exit 1; }

mkdir -p "$OUT"
find "$OUT" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
mkdir -p "$OUT/data/processed/ctr_phase3"
cp "$ROOT/deploy/api/Dockerfile" "$ROOT/deploy/api/render.yaml" "$ROOT/deploy/api/README.md" "$ROOT/requirements-api.txt" "$OUT/"
cp "$ROOT/deploy/api/gitignore" "$OUT/.gitignore"
rsync -a --exclude __pycache__ --exclude .DS_Store "$ROOT/src" "$ROOT/models" "$OUT/"
cp "$ROOT/data/processed/ctr_phase3/manifest.json" "$OUT/data/processed/ctr_phase3/"

[ -d "$OUT/.git" ] || git -C "$OUT" init -q -b main
echo "Listo: $OUT ($(du -sh --exclude=.git "$OUT" 2>/dev/null | cut -f1 || du -sh "$OUT" | cut -f1))"
