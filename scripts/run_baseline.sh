#!/usr/bin/env bash
# Step 2: run the unmodified template once to prove the Docker + OpenFOAM chain works.
# Output: runs/baseline (ignored by git) + a short log summary.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d cases/airfoil_template ] || { echo "Run scripts/fetch_template.sh first"; exit 1; }
mkdir -p runs && rm -rf runs/baseline 2>/dev/null || true
cp -r cases/airfoil_template runs/baseline
./scripts/foam.sh runs/baseline './Allclean >/dev/null 2>&1 || true; ./Allrun; ls; tail -n 25 log.simpleFoam'
echo "Baseline finished: see runs/baseline/log.simpleFoam"
