#!/usr/bin/env bash
# Run OpenFOAM commands inside your Docker image, with a host folder mounted at /work.
# Usage:  scripts/foam.sh <host_dir> "<commands>"
# Image:  auto-detected from `docker images` (first name containing "openfoam"),
#         or set it yourself:  export OPENFOAM_IMAGE=opencfd/openfoam-default:2512
set -euo pipefail
HOST_DIR="$(cd "${1:-.}" && pwd)"; shift || true
CMD="${*:-echo OpenFOAM \$WM_PROJECT_VERSION ready}"
IMAGE="${OPENFOAM_IMAGE:-$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -i openfoam | head -n1 || true)}"
if [ -z "$IMAGE" ]; then
  echo "No OpenFOAM Docker image found. Run 'docker images' and set OPENFOAM_IMAGE=<name:tag>" >&2; exit 1
fi
echo ">> image: $IMAGE | mounted: $HOST_DIR -> /work"
docker run --rm -e OMPI_ALLOW_RUN_AS_ROOT=1 -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 -v "$HOST_DIR":/work -w /work --entrypoint /bin/bash "$IMAGE" -c \
  "for f in /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc; do [ -f \"\$f\" ] && { source \"\$f\"; break; }; done; set -e; $CMD"
