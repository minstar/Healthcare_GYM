FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y bash git curl gettext-base && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get update \
    && apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*

# Pre-install MCP servers — npm registry packages
RUN npm install -g clinicaltrialsgov-mcp-server@1.0.8
RUN npm install -g bach-openfda

# Python MCP servers
RUN uv pip install --system "open-targets-platform-mcp @ git+https://github.com/opentargets/open-targets-platform-mcp.git"

# NPM registry — healthcare bundle (dicom-parser is a peer dep)
RUN npm install -g healthcare-mcp dicom-parser

# Python MCP servers
RUN uv pip install --system biomcp-cli

# Build-from-source MCP servers (Augmented-Nature + NCBI)
RUN git clone --depth 1 https://github.com/Augmented-Nature/ChEMBL-MCP-Server.git /opt/mcp/chembl \
    && cd /opt/mcp/chembl && npm install && npm run build
RUN git clone --depth 1 https://github.com/Augmented-Nature/Augmented-Nature-UniProt-MCP-Server.git /opt/mcp/uniprot \
    && cd /opt/mcp/uniprot && npm install && npm run build
RUN git clone --depth 1 https://github.com/Augmented-Nature/PubChem-MCP-Server.git /opt/mcp/pubchem \
    && cd /opt/mcp/pubchem && npm install && npm run build
RUN git clone --depth 1 https://github.com/Augmented-Nature/KEGG-MCP-Server.git /opt/mcp/kegg \
    && cd /opt/mcp/kegg && npm install && npm run build
RUN git clone --depth 1 https://github.com/Augmented-Nature/NCBI-Datasets-MCP-Server.git /opt/mcp/ncbi \
    && cd /opt/mcp/ncbi && npm install && npm run build

WORKDIR /agent-environment
COPY pyproject.toml ./
COPY src/ ./src/
COPY entrypoint.sh ./

# Create placeholder config so mcp_client.py can be imported at build time
RUN cp src/agent_environment/mcp_server_template.json \
       src/agent_environment/mcp_server_config.json

RUN uv venv --python 3.12
RUN uv sync

EXPOSE 1984

ENTRYPOINT ["/agent-environment/entrypoint.sh"]
CMD ["uv", "run", "python", "-m", "uvicorn", "agent_environment.main:app", "--host", "0.0.0.0", "--port", "1984"]
