#!/usr/bin/env bash
# Step 1: copy the official OpenFOAM airFoil2D tutorial (NACA 0012, simpleFoam, Spalart-Allmaras)
# out of the Docker image into cases/airfoil_template. Run once.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d cases/airfoil_template ]; then echo "cases/airfoil_template already exists"; exit 0; fi
./scripts/foam.sh . 'cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/airFoil2D" cases/airfoil_template && find cases/airfoil_template -maxdepth 2 | sort'
echo "Template copied to cases/airfoil_template"
