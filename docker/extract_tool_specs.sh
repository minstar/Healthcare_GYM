#!/bin/bash
set -euo pipefail

# Extract actual tool specs from running medical MCP Docker container.
# Run this AFTER docker build to replace the manually-written tool_specs_medical.json.
#
# Usage:
#   ./extract_tool_specs.sh [PORT] [OUTPUT_PATH]
#
# Example:
#   ./extract_tool_specs.sh 6986 ../../dive-synth/artifacts/tool_specs_medical.json

PORT="${1:-6986}"
OUTPUT="${2:-tool_specs_medical.json}"
IMAGE="medical-mcp-env:1.0"
CONTAINER="medical-mcp-extract-$$"

echo "Starting ${IMAGE} on port ${PORT}..."
docker run -d -p "${PORT}:1984" --name "${CONTAINER}" "${IMAGE}"

cleanup() { docker rm -f "${CONTAINER}" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for server..."
WAIT=0
until curl -sS --max-time 3 -X POST "http://localhost:${PORT}/list-tools" \
        -H "Content-Type: application/json" 2>/dev/null | python3 -c "import json,sys; json.load(sys.stdin)" 2>/dev/null; do
    sleep 3; WAIT=$((WAIT+3))
    if [ "${WAIT}" -ge 120 ]; then echo "ERR: timeout"; exit 1; fi
done

echo "Extracting tool specs..."
curl -sS -X POST "http://localhost:${PORT}/list-tools" \
    -H "Content-Type: application/json" \
    | python3 -c "
import json, sys
tools = json.load(sys.stdin)
# Convert MCP Tool objects to the dive-synth tool_cache format
specs = []
for t in tools:
    spec = {
        'name': t['name'],
        'description': t.get('description', ''),
        'inputSchema': t.get('inputSchema', {}),
    }
    specs.append(spec)
print(json.dumps(specs, indent=2, ensure_ascii=False))
" > "${OUTPUT}"

N=$(python3 -c "import json; print(len(json.load(open('${OUTPUT}'))))")
echo "Extracted ${N} tool specs to ${OUTPUT}"
