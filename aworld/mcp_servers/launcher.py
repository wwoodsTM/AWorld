#!/usr/bin/env python
"""
MCP Servers Launcher

This module provides a unified entry point for launching all MCP servers.
It handles port allocation, process management, and graceful shutdown
of multiple MCP server instances.

Key features:
- Concurrent server launching
- Random port allocation
- Process lifecycle management
- Signal handling for graceful shutdown

Main functions:
- main: Entry point for launching all MCP servers
- MCPLauncher class: Manages the lifecycle of MCP server processes
"""
import argparse
import signal
import sys
from typing import List

from mcp.server import FastMCP

from aworld.logs.util import logger
from aworld.mcp_servers import *
from aworld.mcp_servers.abc.base import MCPServerBase


class MCPLauncher:
    """
    Manages the lifecycle of MCP server processes.
    """

    def __init__(self, sse_path="/sse", port=20000):
        """
        Initialize the MCP launcher.

        Args:
            port (int, optional): Port to run the FastMCP server on. Defaults to 2000.
        """
        self.sse_path = sse_path
        self.port = port
        self.server = FastMCP("Aworld MCP Server")
        self.processes = {}
        self.running = False
        self.server_instances = []
        self.available_apis = []

    def register_server(self):
        """Register all available MCP server APIs."""
        self.server_instances: List[MCPServerBase] = [
            cls.get_instance()
            for _, cls in globals().items()
            if isinstance(cls, type)
            and issubclass(cls, MCPServerBase)
            and cls is not MCPServerBase
        ]
        logger.success(f"Registered {len(self.server_instances)} MCP server instances")

    def register_apis(self):
        """Register all available MCP server APIs."""
        self.available_apis = [
            func
            for instance in self.server_instances
            for func in instance.get_functions()
        ]
        logger.success(f"Registered {len(self.available_apis)} MCP server APIs")

    def start(self):
        """Start the FastMCP server with all registered functions."""
        if self.running:
            logger.warning("MCP Launcher is already running")
            return

        self.register_server()
        self.register_apis()

        # Initialize and start the FastMCP server
        for api in self.available_apis:
            self.server.add_tool(api)
        self.server.settings.sse_path = self.sse_path
        self.server.settings.port = self.port
        logger.info(f"MCP Launcher started on port {self.port}:{self.sse_path}")

        self.server.run(transport="sse")
        self.running = True

    def stop(self):
        """Stop the FastMCP server and all running processes."""
        if not self.running:
            logger.warning("MCP Launcher is not running")
            return
        # Stop the FastMCP server
        for instance in self.server_instances:
            instance.cleanup()
        self.running = False

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """
    Main entry point for the MCP Launcher.
    Starts the MCP server and handles graceful shutdown.
    """
    parser = argparse.ArgumentParser(description="Launch MCP servers")
    parser.add_argument(
        "--port", type=int, default=20000, help="Port to run the MCP server on"
    )
    parser.add_argument(
        "--sse-path", type=str, default="/sse", help="SSE path for the MCP server"
    )
    args = parser.parse_args()

    launcher = MCPLauncher(sse_path=args.sse_path, port=args.port)

    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping MCP Launcher...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        launcher.start()
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping MCP Launcher...")


if __name__ == "__main__":
    main()
