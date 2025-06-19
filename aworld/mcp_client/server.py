from __future__ import annotations

import abc
import asyncio
from datetime import timedelta
import logging
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from pathlib import Path
from typing import Any, Literal

from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp import ClientSession, StdioServerParameters, Tool as MCPTool, stdio_client
from mcp.client.sse import sse_client
from mcp.types import CallToolResult, JSONRPCMessage
from typing_extensions import NotRequired, TypedDict


class MCPServer(abc.ABC):
    """Base class for Model Context Protocol servers."""

    @abc.abstractmethod
    async def connect(self):
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A readable name for the server."""
        pass

    @abc.abstractmethod
    async def cleanup(self):
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """
        pass

    @abc.abstractmethod
    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the server."""
        pass

    @abc.abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        """Invoke a tool on the server."""
        pass


class _MCPServerWithClientSession(MCPServer, abc.ABC):
    """Base class for MCP servers that use a `ClientSession` to communicate with the server."""

    def __init__(self, cache_tools_list: bool, session_connect_timeout_seconds: int = 30):
        """
        Args:
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
            cached and only fetched from the server once. If `False`, the tools list will be
            fetched from the server on each call to `list_tools()`. The cache can be invalidated
            by calling `invalidate_tools_cache()`. You should set this to `True` if you know the
            server will not change its tools list, because it can drastically improve latency
            (by avoiding a round-trip to the server every time).

            session_connect_timeout_seconds: session connect timeout seconds
        """
        self.session: ClientSession | None = None
        self.exit_stack: AsyncExitStack = AsyncExitStack()
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.cache_tools_list = cache_tools_list
        self.session_connect_timeout_seconds = timedelta(seconds=session_connect_timeout_seconds)

        # The cache is always dirty at startup, so that we fetch tools at least once
        self._cache_dirty = True
        self._tools_list: list[MCPTool] | None = None

    @abc.abstractmethod
    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        pass

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    def invalidate_tools_cache(self):
        """Invalidate the tools cache."""
        self._cache_dirty = True

    async def connect(self):
        """Connect to the server."""
        try:
            # Ensure closing previous exit_stack to avoid nested async contexts
            if hasattr(self, 'exit_stack') and self.exit_stack:
                try:
                    await self.exit_stack.aclose()
                except Exception as e:
                    logging.error(f"Error closing previous exit stack: {e}")

            self.exit_stack = AsyncExitStack()

            # Use a single task context to create the connection
            transport = await self.exit_stack.enter_async_context(self.create_streams())
            read, write = transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=self.session_connect_timeout_seconds))
            await session.initialize()
            self.session = session
        except Exception as e:
            logging.error(f"Error initializing MCP server: {e}")
            # Ensure resources are cleaned up if connection fails
            await self.cleanup()
            raise

    async def list_tools(self) -> list[MCPTool]:
        """List the tools available on the server."""
        if not self.session:
            raise RuntimeError("Server not initialized. Make sure you call `connect()` first.")

        # Return from cache if caching is enabled, we have tools, and the cache is not dirty
        if self.cache_tools_list and not self._cache_dirty and self._tools_list:
            return self._tools_list

        # Reset the cache dirty to False
        self._cache_dirty = False

        # Fetch the tools from the server
        self._tools_list = (await self.session.list_tools()).tools
        return self._tools_list

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        """Invoke a tool on the server."""
        if not self.session:
            raise RuntimeError("Server not initialized. Make sure you call `connect()` first.")

        return await self.session.call_tool(tool_name, arguments)

    async def cleanup(self):
        """Cleanup the server."""
        async with self._cleanup_lock:
            try:
                # Ensure cleanup operations occur in the same task context
                session = self.session
                self.session = None  # Remove reference first

                # Wait briefly to ensure any pending operations complete
                try:
                    await asyncio.sleep(0.1)
                except asyncio.CancelledError:
                    # Ignore cancellation exceptions, continue cleaning resources
                    pass

                # Clean up exit_stack, ensuring all resources are properly closed
                exit_stack = self.exit_stack
                if exit_stack:
                    try:
                        await exit_stack.aclose()
                    except Exception as e:
                        logging.error(f"Error closing exit stack during cleanup: {e}")
            except Exception as e:
                logging.error(f"Error during server cleanup: {e}")


class MCPServerStdioParams(TypedDict):
    """Mirrors `mcp.client.stdio.StdioServerParameters`, but lets you pass params without another
    import.
    """

    command: str
    """The executable to run to start the server. For example, `python` or `node`."""

    args: NotRequired[list[str]]
    """Command line args to pass to the `command` executable. For example, `['foo.py']` or
    `['server.js', '--port', '8080']`."""

    env: NotRequired[dict[str, str]]
    """The environment variables to set for the server. ."""

    cwd: NotRequired[str | Path]
    """The working directory to use when spawning the process."""

    encoding: NotRequired[str]
    """The text encoding used when sending/receiving messages to the server. Defaults to `utf-8`."""

    encoding_error_handler: NotRequired[Literal["strict", "ignore", "replace"]]
    """The text encoding error handler. Defaults to `strict`.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values.
    """


class MCPServerStdio(_MCPServerWithClientSession):
    """MCP server implementation that uses the stdio transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#stdio) for
    details.
    """

    def __init__(
            self,
            params: MCPServerStdioParams,
            cache_tools_list: bool = False,
            name: str | None = None,
    ):
        """Create a new MCP server based on the stdio transport.

        Args:
            params: The params that configure the server. This includes the command to run to
                start the server, the args to pass to the command, the environment variables to
                set for the server, the working directory to use when spawning the process, and
                the text encoding used when sending/receiving messages to the server.
            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).
            name: A readable name for the server. If not provided, we'll create one from the
                command.
        """
        super().__init__(cache_tools_list, int(params.get("env").get("SESSION_REQUEST_CONNECT_TIMEOUT", "60")))

        self.params = StdioServerParameters(
            command=params["command"],
            args=params.get("args", []),
            env=params.get("env"),
            cwd=params.get("cwd"),
            encoding=params.get("encoding", "utf-8"),
            encoding_error_handler=params.get("encoding_error_handler", "strict"),
        )

        self._name = name or f"stdio: {self.params.command}"

    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        return stdio_client(self.params)

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name


class MCPServerSseParams(TypedDict):
    """Mirrors the params in`mcp.client.sse.sse_client`."""

    url: str
    """The URL of the server."""

    headers: NotRequired[dict[str, str]]
    """The headers to send to the server."""

    timeout: NotRequired[float]
    """The timeout for the HTTP request. Defaults to 60 seconds."""

    sse_read_timeout: NotRequired[float]
    """The timeout for the SSE connection, in seconds. Defaults to 5 minutes."""


class MCPServerSse(_MCPServerWithClientSession):
    """MCP server implementation that uses the HTTP with SSE transport. See the [spec]
    (https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/#http-with-sse)
    for details.
    """

    def __init__(
            self,
            params: MCPServerSseParams,
            cache_tools_list: bool = False,
            name: str | None = None,
    ):
        """Create a new MCP server based on the HTTP with SSE transport.

        Args:
            params: The params that configure the server. This includes the URL of the server,
                the headers to send to the server, the timeout for the HTTP request, and the
                timeout for the SSE connection.

            cache_tools_list: Whether to cache the tools list. If `True`, the tools list will be
                cached and only fetched from the server once. If `False`, the tools list will be
                fetched from the server on each call to `list_tools()`. The cache can be
                invalidated by calling `invalidate_tools_cache()`. You should set this to `True`
                if you know the server will not change its tools list, because it can drastically
                improve latency (by avoiding a round-trip to the server every time).

            name: A readable name for the server. If not provided, we'll create one from the
                URL.
        """
        super().__init__(cache_tools_list)

        self.params = params
        self._name = name or f"sse: {self.params['url']}"

    def create_streams(
            self,
    ) -> AbstractAsyncContextManager[
        tuple[
            MemoryObjectReceiveStream[JSONRPCMessage | Exception],
            MemoryObjectSendStream[JSONRPCMessage],
        ]
    ]:
        """Create the streams for the server."""
        return sse_client(
            url=self.params["url"],
            headers=self.params.get("headers", None),
            timeout=self.params.get("timeout", 60),
            sse_read_timeout=self.params.get("sse_read_timeout", 60 * 5),
        )

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self._name
