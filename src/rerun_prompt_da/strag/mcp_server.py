"""
MCP Server — FastAPI JSON-RPC server exposing agent tools over HTTP.

Replicates dimos/agents/mcp/mcp_server.py pattern, standalone.

The MCP (Model Context Protocol) server:
  1. Lists available tools via tools/list → JSON-RPC
  2. Executes tool calls via tools/call → JSON-RPC
  3. Agent (OTAAgent / any LangGraph client) fetches tools on startup,
     then calls them via HTTP without needing direct Python imports.

This lets you decouple the LLM agent from the robot control code:
  - Robot process: runs MCP server exposing navigate_to, stop, tag_location, etc.
  - Agent process: connects to MCP server, fetches tools, makes decisions

DIMENSIONAL'S PATTERN:
  McpServer runs as a Module (forkserver worker).
  McpClient connects via HTTP and uses the tools as LangChain StructuredTools.
  The server auto-discovers @skill-decorated methods from deployed modules.
  Here we just register tools explicitly.

Usage:
    from rerun_prompt_da.strag.mcp_server import MCPServer, tool
    from rerun_prompt_da.strag.ota_agent import AgentTools, NarrativeLedger

    ledger = NarrativeLedger()
    agent_tools = AgentTools(ledger, rag_client=rag, navigate_fn=robot.nav_to, ...)

    server = MCPServer(port=9990)
    server.register_tools_from(agent_tools)
    server.run()  # blocks; or server.run_in_background()
"""
from __future__ import annotations

import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool descriptor
# ---------------------------------------------------------------------------

@dataclass
class ToolDescriptor:
    name: str
    description: str
    input_schema: dict
    fn: Callable


def _build_input_schema(fn: Callable) -> dict:
    """Build a JSON Schema for the function's parameters."""
    sig = inspect.signature(fn)
    props: dict[str, dict] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = param.annotation
        json_type = type_map.get(ann, "string")
        props[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

        # Extract description from docstring param section
        doc = fn.__doc__ or ""
        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped.startswith(f"{name}:") or stripped.startswith(f"{name} ("):
                desc = stripped.split(":", 1)[-1].strip()
                props[name]["description"] = desc
                break

    return {
        "type": "object",
        "properties": props,
        "required": required,
    }


def _extract_description(fn: Callable) -> str:
    doc = fn.__doc__ or ""
    lines = doc.strip().split("\n")
    # First non-empty line is the description
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("Args:") and not stripped.startswith("Returns:"):
            return stripped
    return fn.__name__


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class MCPServer:
    """
    Lightweight MCP (Model Context Protocol) server using FastAPI.

    Implements:
      POST /mcp → JSON-RPC 2.0
        - initialize: handshake
        - tools/list: return all registered tools
        - tools/call: execute a tool
    """

    MCP_VERSION = "2025-11-25"

    def __init__(self, host: str = "0.0.0.0", port: int = 9990) -> None:
        self._host = host
        self._port = port
        self._tools: dict[str, ToolDescriptor] = {}
        self._app = None
        self._server_thread: Optional[threading.Thread] = None

    def register(
        self,
        fn: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a function as an MCP tool."""
        tool_name = name or fn.__name__
        desc = description or _extract_description(fn)
        schema = _build_input_schema(fn)
        self._tools[tool_name] = ToolDescriptor(
            name=tool_name,
            description=desc,
            input_schema=schema,
            fn=fn,
        )
        logger.info(f"Registered tool: {tool_name}")

    def register_tools_from(self, obj: Any, names: Optional[list[str]] = None) -> None:
        """
        Register all public methods of an object as tools.
        If names is provided, only register those methods.
        Methods starting with '_' are skipped.
        """
        methods = names or [
            n for n in dir(obj)
            if not n.startswith("_") and callable(getattr(obj, n))
        ]
        for method_name in methods:
            method = getattr(obj, method_name, None)
            if method and callable(method):
                try:
                    self.register(method)
                except Exception as e:
                    logger.warning(f"Skipped {method_name}: {e}")

    def _build_app(self):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI(title="MCP Server")

        @app.post("/mcp")
        async def mcp_endpoint(request: Request):
            body = await request.json()
            response = self._handle_rpc(body)
            return JSONResponse(content=response)

        return app

    def _handle_rpc(self, body: dict) -> dict:
        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params") or {}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": self.MCP_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "prompt-da-mcp-server", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": [self._tool_to_mcp(t) for t in self._tools.values()]}
            elif method == "tools/call":
                result = self._call_tool(params.get("name", ""), params.get("arguments", {}))
            else:
                return self._error(rpc_id, -32601, f"Method not found: {method}")
        except Exception as e:
            logger.error(f"RPC error: {e}", exc_info=True)
            return self._error(rpc_id, -32603, str(e))

        return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

    def _tool_to_mcp(self, tool: ToolDescriptor) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": tool.input_schema,
        }

    def _call_tool(self, name: str, arguments: dict) -> dict:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")
        try:
            result = tool.fn(**arguments)
        except TypeError as e:
            raise ValueError(f"Invalid arguments for {name}: {e}") from e

        if result is None:
            text = "Done."
        elif isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, default=str)

        return {"content": [{"type": "text", "text": text}]}

    @staticmethod
    def _error(rpc_id: Any, code: int, message: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": code, "message": message},
        }

    def run(self, log_level: str = "warning") -> None:
        """Start the server (blocks)."""
        import uvicorn
        self._app = self._build_app()
        uvicorn.run(self._app, host=self._host, port=self._port, log_level=log_level)

    def run_in_background(self, log_level: str = "warning") -> None:
        """Start the server in a background thread (non-blocking)."""
        def _run():
            self.run(log_level=log_level)

        self._server_thread = threading.Thread(target=_run, daemon=True, name="mcp-server")
        self._server_thread.start()
        logger.info(f"MCP server running at http://{self._host}:{self._port}/mcp")

    def url(self) -> str:
        return f"http://{self._host}:{self._port}/mcp"


# ---------------------------------------------------------------------------
# MCP Client (connects to any MCP server and fetches tools as LangChain tools)
# ---------------------------------------------------------------------------

class MCPClient:
    """
    HTTP client that fetches tools from an MCP server and converts them to
    LangChain StructuredTool objects for use with LangGraph agents.

    This is the client side — connects to MCPServer running in another process.
    """

    def __init__(self, server_url: str = "http://localhost:9990/mcp") -> None:
        self._url = server_url
        import httpx
        self._http = httpx.Client(timeout=120.0)
        self._seq = 0

    def _rpc(self, method: str, params: Optional[dict] = None) -> dict:
        self._seq += 1
        body = {"jsonrpc": "2.0", "id": self._seq, "method": method}
        if params:
            body["params"] = params
        resp = self._http.post(self._url, json=body)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']['message']}")
        return data.get("result", {})

    def initialize(self) -> dict:
        return self._rpc("initialize")

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        parts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        return "\n".join(parts)

    def as_langchain_tools(self, timeout: float = 30.0) -> list:
        """Fetch tools from server and return as LangChain StructuredTools."""
        from langchain_core.tools import StructuredTool
        import time

        deadline = time.monotonic() + timeout
        while True:
            try:
                self.initialize()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"MCP server not reachable at {self._url}")
                time.sleep(1.0)

        raw_tools = self.list_tools()
        lc_tools = []
        for t in raw_tools:
            name = t["name"]
            description = t.get("description", "")
            schema = t.get("inputSchema", {"type": "object", "properties": {}})

            def make_call(tool_name=name):
                def call_fn(**kwargs):
                    return self.call_tool(tool_name, kwargs)
                call_fn.__name__ = tool_name
                return call_fn

            lc_tools.append(StructuredTool(
                name=name,
                description=description,
                func=make_call(),
                args_schema=schema,
            ))
        return lc_tools

    def close(self) -> None:
        self._http.close()


# ---------------------------------------------------------------------------
# Convenience: build a full MCP server from AgentTools in one call
# ---------------------------------------------------------------------------

def make_mcp_server(
    agent_tools,
    port: int = 9990,
    host: str = "0.0.0.0",
    extra_tool_names: Optional[list[str]] = None,
) -> MCPServer:
    """
    Build and return an MCPServer pre-loaded with AgentTools.

    Args:
        agent_tools: AgentTools instance from ota_agent.py
        port: HTTP port to listen on
        host: Host to bind
        extra_tool_names: Additional method names to register (default: standard set)

    Example:
        server = make_mcp_server(agent_tools, port=9990)
        server.run_in_background()
    """
    default_tools = [
        "think", "navigate_to", "navigate_to_location", "stop",
        "tag_location", "query_memory", "get_ledger", "update_ledger",
        "get_spatial_context",
    ]
    tool_names = (extra_tool_names or []) + default_tools

    server = MCPServer(host=host, port=port)
    for name in tool_names:
        fn = getattr(agent_tools, name, None)
        if fn and callable(fn):
            server.register(fn)

    return server
