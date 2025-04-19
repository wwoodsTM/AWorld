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
import random

from mcp.server import FastMCP

from aworld.logs.util import logger
from aworld.mcp_servers.arxiv_server import ArxivServer
from aworld.mcp_servers.audio_server import AudioServer
from aworld.mcp_servers.code_server import CodeServer
from aworld.mcp_servers.document_server import DocumentServer
from aworld.mcp_servers.download_server import DownloadServer
from aworld.mcp_servers.github_server import GitHubServer
from aworld.mcp_servers.googlemaps_server import GoogleMapsServer
from aworld.mcp_servers.image_server import ImageServer
from aworld.mcp_servers.math_server import MathServer
from aworld.mcp_servers.reasoning_server import ReasoningServer
from aworld.mcp_servers.reddit_server import RedditServer
from aworld.mcp_servers.search_server import SearchServer
from aworld.mcp_servers.video_server import VideoServer


class MCPLauncher:
    """
    Manages the lifecycle of MCP server processes.
    """

    def __init__(self, sse_path="/sse", port=2000):
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

    def register_apis(self):
        """Register all available MCP server APIs."""
        arxiv_server = ArxivServer.get_instance()
        audio_server = AudioServer.get_instance()
        code_server = CodeServer.get_instance()
        document_server = DocumentServer.get_instance()
        download_server = DownloadServer.get_instance()
        github_server = GitHubServer.get_instance()
        googlemaps_server = GoogleMapsServer.get_instance()
        image_server = ImageServer.get_instance()
        math_server = MathServer.get_instance()
        reasoning_server = ReasoningServer.get_instance()
        reddit_server = RedditServer.get_instance()
        search_server = SearchServer.get_instance()
        video_server = VideoServer.get_instance()

        self.server_instances = [
            arxiv_server,
            audio_server,
            code_server,
            document_server,
            download_server,
            github_server,
            googlemaps_server,
            image_server,
            math_server,
            reasoning_server,
            reddit_server,
            search_server,
            video_server,
        ]

        self.available_apis = [
            arxiv_server.search_arxiv_paper_by_title_or_ids,
            arxiv_server.download_arxiv_paper,
            audio_server.transcribe_audio,
            code_server.generate_code,
            code_server.execute_code,
            document_server.read_text,
            document_server.read_json,
            document_server.read_xml,
            document_server.read_pdf,
            document_server.read_docx,
            document_server.read_excel,
            document_server.read_pptx,
            document_server.read_source_code,
            document_server.read_html_text,
            download_server.download_files,
            github_server.get_repository,
            github_server.list_repository_contents,
            github_server.search_repositories,
            github_server.search_code,
            github_server.get_user,
            github_server.get_user_repositories,
            github_server.get_labels,
            github_server.get_issues,
            github_server.get_file_content,
            googlemaps_server.geocode,
            googlemaps_server.distance_matrix,
            googlemaps_server.directions,
            googlemaps_server.place_details,
            googlemaps_server.place_search,
            googlemaps_server.get_latlng,
            googlemaps_server.get_postcode,
            image_server.ocr,
            image_server.reasoning_image,
            math_server.basic_math,
            math_server.statistics,
            math_server.geometry,
            math_server.trigonometry,
            math_server.solve_equation,
            math_server.random_operations,
            math_server.unit_conversion,
            reasoning_server.complex_problem_reasoning,
            reddit_server.get_hot_posts,
            reddit_server.search_reddit,
            reddit_server.get_post_comments,
            reddit_server.get_subreddit_info,
            reddit_server.get_user_info,
            reddit_server.get_user_posts,
            reddit_server.get_top_subreddits,
            search_server.search_google,
            # search_server.search_duckduckgo,
            # search_server.search_exa,
            video_server.analyze_video,
            video_server.extract_video_subtitles,
            video_server.summarize_video,
        ]

    def start(self):
        """Start the FastMCP server with all registered functions."""
        if self.running:
            logger.warning("MCP Launcher is already running")
            return

        self.register_apis()

        # Initialize and start the FastMCP server
        for api in self.available_apis:
            self.server.add_tool(api)
        self.server.settings.sse_path = self.sse_path
        self.server.settings.port = self.port
        self.server.run(transport="sse")
        self.running = True
        logger.info(
            f"MCP Launcher started on port {self.port}/{self.sse_path}"
            f"with {len(self.available_apis)} total functions"
        )

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
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="Launch MCP servers")
    parser.add_argument(
        "--port", type=int, default=2000, help="Port to run the MCP server on"
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
        # Keep the main thread alive
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping MCP Launcher...")


if __name__ == "__main__":
    main()
