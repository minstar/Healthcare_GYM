import contextlib
import mcp
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mcp.types
from .mcp_client import client, config
from .logger import create_logger
from cacheout import Cache
import json
import hashlib
import random

CACHE_TTL_HOURS = 48

logger = create_logger(__name__)

tool_cache = Cache(
    maxsize=10000,
    ttl=CACHE_TTL_HOURS * 60 * 60,
    enable_stats=True,
)

CACHEABLE_SERVERS = {
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
}


class CallToolRequest(BaseModel):
    tool_name: str
    tool_args: Dict[str, Any]
    use_cache: bool = True


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    mcp_servers = config.get("mcpServers", {})
    logger.info(
        f"Starting medical MCP environment with {len(mcp_servers)} servers: {list(mcp_servers.keys())}"
    )
    async with client:
        tools = await client.list_tools()
        logger.info(f"{len(tools)} tools loaded in total")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Medical MCP Environment API"}


@app.post("/list-tools")
async def list_tools() -> list[mcp.types.Tool]:
    async with client:
        try:
            return await client.list_tools()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to list tools: {str(e)}"
            )


def should_cache_tool(tool_name: str) -> bool:
    server_name = tool_name.split("_", 1)[0]
    return server_name in CACHEABLE_SERVERS


def generate_cache_key(tool_name: str, tool_args: dict) -> str:
    cache_data = {"tool_name": tool_name, "tool_args": tool_args}
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.md5(cache_str.encode()).hexdigest()


@app.post("/call-tool")
async def call_tool(
    request: CallToolRequest,
) -> list[mcp.types.ContentBlock]:
    cache_key = generate_cache_key(request.tool_name, request.tool_args)

    cached_result = tool_cache.get(cache_key)
    if (
        cached_result is not None
        and request.use_cache
        and should_cache_tool(request.tool_name)
    ):
        logger.info(f"Returning cached result for tool '{request.tool_name}'")
        return cached_result

    async with client:
        try:
            result = await client.call_tool(request.tool_name, request.tool_args)

            if result.is_error:
                error_msg = "Unknown error"
                if result.content and isinstance(
                    result.content[0], mcp.types.TextContent
                ):
                    error_msg = result.content[0].text
                raise HTTPException(
                    status_code=500,
                    detail=f"Tool '{request.tool_name}' execution failed: {error_msg}",
                )

            content_blocks = result.content
            if should_cache_tool(request.tool_name) and cache_key is not None:
                random_ttl = int(CACHE_TTL_HOURS * 60 * 60 * random.uniform(0.7, 1.0))
                tool_cache.set(cache_key, content_blocks, ttl=random_ttl)

            return content_blocks

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to call tool '{request.tool_name}': {str(e)}",
            )


@app.post("/reset-state")
async def reset_state():
    tool_cache.clear()
    return {"message": "State reset (cache cleared)"}


@app.get("/cache-stats")
async def get_cache_stats():
    return {
        "cache_size": len(tool_cache),
        "max_size": tool_cache.maxsize,
        "ttl_seconds": tool_cache.ttl,
    }
