from fastmcp import Client
from fastmcp.client.logging import LogMessage
from .logger import create_logger
import json
import os
from pathlib import Path

logger = create_logger(__name__)

template_path = Path(__file__).parent / "mcp_server_template.json"
with open(template_path) as f:
    template_config = json.load(f)

config_path = Path(__file__).parent / "mcp_server_config.json"
with open(config_path) as f:
    config = json.load(f)

DEFAULT_SERVERS = [
    "clinicaltrialsgov-mcp-server",
    "pubmed",
    "openfda",
    "opentargets",
    "chembl",
    "uniprot",
    "pubchem",
    "kegg",
    "ncbi-datasets",
    "healthcare",
    "biomcp",
]

enabled_servers = os.getenv("ENABLED_SERVERS", "").strip()

if "mcpServers" in config:
    if enabled_servers:
        enabled_list = [s.strip() for s in enabled_servers.split(",")]
        enabled_set = set(enabled_list)
    else:
        enabled_set = set(DEFAULT_SERVERS)

    config["mcpServers"] = {
        name: server_config
        for name, server_config in config["mcpServers"].items()
        if name in enabled_set
    }
    logger.info(f"Enabled {len(config['mcpServers'])} servers: {list(config['mcpServers'].keys())}")


async def log_handler(message: LogMessage) -> None:
    level = message.level.upper()
    data = message.data
    match level:
        case "debug":
            logger.debug(data)
        case "info":
            logger.info(data)
        case "warning":
            logger.warning(data)
        case "error":
            logger.error(data)
        case _:
            logger.info(data)


client: Client = Client(
    config,
    log_handler=log_handler,
)
