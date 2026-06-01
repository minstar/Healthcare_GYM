#!/bin/bash
set -euo pipefail

TAG="${1:-medical-mcp-env:1.0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/images"
mkdir -p "${OUT_DIR}"

echo "Building ${TAG} ..."
docker build -t "${TAG}" "${SCRIPT_DIR}"

TAR_NAME="$(echo "${TAG}" | sed 's/:/-/')-amd64.tar"
echo "Saving to ${OUT_DIR}/${TAR_NAME} ..."
docker save "${TAG}" -o "${OUT_DIR}/${TAR_NAME}"

echo "Done: ${OUT_DIR}/${TAR_NAME}"
ls -lh "${OUT_DIR}/${TAR_NAME}"
