#!/bin/bash
set -e

CONFIG_DIR="/agent-environment/src/agent_environment"

# Generate mcp_server_config.json from template (envsubst for API keys)
if [ -f "${CONFIG_DIR}/mcp_server_template.json" ]; then
    envsubst < "${CONFIG_DIR}/mcp_server_template.json" > "${CONFIG_DIR}/mcp_server_config.json"
fi

exec "$@"
